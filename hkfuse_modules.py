"""Core modules for HKFuse.

This file contains the RoPE helper used by the bottleneck inter-transformer,
the bidirectional KDA operator wrapper, HK-Block fusion, and token self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# External KDA kernel dependency.
try:
    from fla.ops.kda import chunk_kda
except ImportError:
    raise ImportError("Please install or expose the FLA package that provides `fla.ops.kda.chunk_kda`.")

# Hilbert ordering dependency.
try:
    from hilbertcurve.hilbertcurve import HilbertCurve
except ImportError:
    HilbertCurve = None
    print("!!! [Warning] 'hilbertcurve' missing.")


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs in the last dimension: [x0, x1] -> [-x1, x0]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _rope_apply_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply one-axis RoPE to x with shape [B, heads, N, dim]."""
    if x.shape[-1] == 0:
        return x
    cos = cos.to(device=x.device, dtype=x.dtype).view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.to(device=x.device, dtype=x.dtype).view(1, 1, sin.shape[0], sin.shape[1])
    return (x * cos) + (_rope_rotate_half(x) * sin)


def _rope_build_cos_sin_1d(
    pos_1d: torch.Tensor,
    dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables for one coordinate axis."""
    if dim == 0:
        n = pos_1d.numel()
        return (
            torch.zeros(n, 0, device=pos_1d.device, dtype=dtype),
            torch.zeros(n, 0, device=pos_1d.device, dtype=dtype),
        )
    assert dim % 2 == 0, f"[RoPE] dim must be even, got dim={dim}"
    device = pos_1d.device
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    angles = pos_1d.to(dtype).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
    return cos, sin


class RotaryPositionalEmbedding3D(nn.Module):
    """**3D RoPE that provides spatial orientation for Hilbert-ordered tokens / 为 Hilbert 排序后的 token 提供空间方向信息的 3D RoPE**"""

    def __init__(
        self,
        spatial_shape: tuple[int, int, int],
        head_dim: int,
        base: float = 10000.0,
        cache_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        D, H, W = spatial_shape
        self.spatial_shape = (D, H, W)
        self.head_dim = head_dim
        self.base = base

        assert head_dim % 2 == 0, f"[RoPE] head_dim must be even, got {head_dim}"
        if head_dim >= 6:
            dz = 2
            dy = 2
            dx = head_dim - dz - dy
            if dx <= 0 or dx % 2 != 0:
                raise ValueError(f"[RoPE] invalid dimension split: dx={dx} (head_dim={head_dim})")
        else:
            dz = 0
            dy = 0
            dx = head_dim
        assert dz % 2 == 0 and dy % 2 == 0 and dx % 2 == 0
        assert dx > 0, "[RoPE] at least one effective x-axis dimension is required"
        assert dz + dy + dx == head_dim

        self.dz, self.dy, self.dx = dz, dy, dx

        z = torch.arange(D).view(D, 1, 1).expand(D, H, W).reshape(-1)
        y = torch.arange(H).view(1, H, 1).expand(D, H, W).reshape(-1)
        x = torch.arange(W).view(1, 1, W).expand(D, H, W).reshape(-1)
        self.register_buffer("_pos_z", z, persistent=False)
        self.register_buffer("_pos_y", y, persistent=False)
        self.register_buffer("_pos_x", x, persistent=False)

        self.cache_dtype = cache_dtype
        self._cached_device = None
        self.register_buffer("_cos_z", torch.empty(0), persistent=False)
        self.register_buffer("_sin_z", torch.empty(0), persistent=False)
        self.register_buffer("_cos_y", torch.empty(0), persistent=False)
        self.register_buffer("_sin_y", torch.empty(0), persistent=False)
        self.register_buffer("_cos_x", torch.empty(0), persistent=False)
        self.register_buffer("_sin_x", torch.empty(0), persistent=False)

    def _build_cache_if_needed(self, device: torch.device):
        if self._cached_device == device and self._cos_z.numel() > 0:
            return

        cos_z, sin_z = _rope_build_cos_sin_1d(
            self._pos_z.to(device), self.dz, base=self.base, dtype=self.cache_dtype
        )
        cos_y, sin_y = _rope_build_cos_sin_1d(
            self._pos_y.to(device), self.dy, base=self.base, dtype=self.cache_dtype
        )
        cos_x, sin_x = _rope_build_cos_sin_1d(
            self._pos_x.to(device), self.dx, base=self.base, dtype=self.cache_dtype
        )

        self._cos_z.resize_(cos_z.shape).copy_(cos_z)
        self._sin_z.resize_(sin_z.shape).copy_(sin_z)
        self._cos_y.resize_(cos_y.shape).copy_(cos_y)
        self._sin_y.resize_(sin_y.shape).copy_(sin_y)
        self._cos_x.resize_(cos_x.shape).copy_(cos_x)
        self._sin_x.resize_(sin_x.shape).copy_(sin_x)
        self._cached_device = device

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert q.shape == k.shape, "[RoPE] q/k shapes must match"
        _, _, n, dh = q.shape
        D, H, W = self.spatial_shape
        assert n == D * H * W, f"[RoPE] N must equal D*H*W, got N={n}, D*H*W={D*H*W}"
        assert dh == self.head_dim, f"[RoPE] head_dim mismatch: expected {self.head_dim}, got {dh}"

        self._build_cache_if_needed(q.device)

        qz, qy, qx = q[..., :self.dz], q[..., self.dz:self.dz + self.dy], q[..., self.dz + self.dy:]
        kz, ky, kx = k[..., :self.dz], k[..., self.dz:self.dz + self.dy], k[..., self.dz + self.dy:]

        if self.dz:
            qz = _rope_apply_1d(qz, self._cos_z, self._sin_z)
            kz = _rope_apply_1d(kz, self._cos_z, self._sin_z)
        if self.dy:
            qy = _rope_apply_1d(qy, self._cos_y, self._sin_y)
            ky = _rope_apply_1d(ky, self._cos_y, self._sin_y)
        if self.dx:
            qx = _rope_apply_1d(qx, self._cos_x, self._sin_x)
            kx = _rope_apply_1d(kx, self._cos_x, self._sin_x)

        return torch.cat([qz, qy, qx], dim=-1), torch.cat([kz, ky, kx], dim=-1)
    
class BiDirectionalChunkKDA(nn.Module):
    """**Bidirectional KDA wrapper with explicit numerical safeguards / 带显式数值保护的双向 KDA 封装**"""

    def __init__(self, dim, head_dim=64, num_heads=None):
        super().__init__()
        self.dim = dim
        
        # **Choose head count from channel width when it is not provided / 未指定时根据通道宽度确定 head 数**
        if num_heads is None:
            if dim < head_dim:
                self.head_dim = dim
                self.num_heads = 1
            else:
                self.head_dim = head_dim
                self.num_heads = dim // head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
        
        assert self.num_heads * self.head_dim == self.dim
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.f_a_proj = nn.Linear(dim, dim, bias=False)
        self.f_b_proj = nn.Linear(dim, dim, bias=False)
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.A_log = nn.Parameter(torch.randn(self.num_heads, self.head_dim).uniform_(-4, -1))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        
        self.group_norm = nn.LayerNorm(dim)

    def forward_kda_core(self, q, k, v, g, beta):
        # **Run the recurrent KDA kernel in FP32 and bound the gate terms to reduce overflow risk / 使用 FP32 执行 KDA 内核并约束 gate 项，降低溢出风险**
        original_dtype = q.dtype

        q, k, v, g, beta = [t.float() for t in (q, k, v, g, beta)]

        # **Normalize Q/K to limit dot-product energy on long token sequences / 对 Q/K 做归一化以限制长序列 dot-product 能量**
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # **Clamp gates away from unstable extremes / 将 gate 限制在更稳定的范围内**
        g = torch.clamp(g, max=-1.0)

        # **Keep beta away from exact 0/1 to avoid saturated gradients / 避免 beta 贴近 0 或 1 导致梯度饱和**
        beta = torch.clamp(beta, min=1e-4, max=1.0 - 1e-4)

        # **Disable autocast around chunk_kda so accumulation remains in FP32 / 在 chunk_kda 周围关闭 autocast，使累积保持 FP32**
        with torch.amp.autocast('cuda', enabled=False):
            out, _ = chunk_kda(q, k, v, g, beta, scale=self.head_dim ** -0.5)
            out = out.float()

        # **Finite-value guard prevents rare kernel anomalies from propagating through the network / 有限值保护避免少量内核异常继续传播**
        if not torch.isfinite(out).all():
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = torch.clamp(out, min=-1e4, max=1e4)
        
        # **Return FP32 to let subsequent normalization/projection handle casting safely / 返回 FP32，由后续归一化和投影安全处理精度**
        return out

    def forward(self, x):
        B, L, C = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        # **Gate generation for KDA recurrence / KDA 递推的 gate 生成**
        gate_feat = self.f_b_proj(self.f_a_proj(x)).view(B, L, self.num_heads, self.head_dim)
        raw_gate = gate_feat + self.dt_bias + self.A_log
        g = F.logsigmoid(raw_gate)
        beta = self.b_proj(x).view(B, L, self.num_heads).sigmoid()

        # **Forward scan / 正向扫描**
        out_fwd = self.forward_kda_core(q, k, v, g, beta)
        
        # **Backward scan by reversing the token sequence / 反转 token 序列执行反向扫描**
        q_rev, k_rev, v_rev = q.flip(1), k.flip(1), v.flip(1)
        g_rev, beta_rev = g.flip(1), beta.flip(1)
        
        out_bwd = self.forward_kda_core(q_rev, k_rev, v_rev, g_rev, beta_rev).flip(1)
        
        out = out_fwd + out_bwd
        
        out = rearrange(out, 'b l h d -> b l (h d)')
        out = self.group_norm(out)
        out = self.o_proj(out)
        return out

class HKBlock(nn.Module):
    """**HK-Block for fixed-slot cross-modal fusion with Hilbert ordering and KDA / 使用固定模态槽位、Hilbert 排序和 KDA 的跨模态融合块**"""

    def __init__(self, dim, spatial_size=None, enable_valid_sort: bool = True):
        super().__init__()
        self.dim = dim
        # **Current HKFuse instantiates HKBlock with enable_valid_sort=False, so modality slots remain fixed / 当前 HKFuse 中该项为 False，模态槽位保持固定**
        self.enable_valid_sort = enable_valid_sort
        self.modality_embed = nn.Parameter(torch.randn(1, 4, 1, dim) * 0.02)
        self.kda_norm = nn.LayerNorm(dim)
        self.short_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GroupNorm(max(1, dim // 8), dim),
            nn.SiLU()
        )
        self.kda = BiDirectionalChunkKDA(dim)
        self.register_buffer("hilbert_idx", torch.zeros(0).long())
        self.register_buffer("inverse_hilbert_idx", torch.zeros(0).long())
        self.current_spatial_shape = (0, 0, 0)

    def _generate_3d_hilbert_curve(self, D, H, W, device):
        if HilbertCurve is None: raise ImportError("No hilbertcurve lib")
        max_side = max(D, H, W)
        p = math.ceil(math.log2(max_side)) if max_side > 0 else 1
        hc = HilbertCurve(p, 3)
        zs = torch.arange(D, device=device).view(D, 1, 1).expand(D, H, W)
        ys = torch.arange(H, device=device).view(1, H, 1).expand(D, H, W)
        xs = torch.arange(W, device=device).view(1, 1, W).expand(D, H, W)
        coords = torch.stack((zs, ys, xs), dim=-1).reshape(-1, 3)
        distances = torch.tensor(hc.distances_from_points(coords.cpu().numpy()), device=device)
        hilbert_idx = torch.argsort(distances)
        inverse_hilbert_idx = torch.argsort(hilbert_idx)
        return hilbert_idx, inverse_hilbert_idx

    def _update_hilbert_cache(self, D, H, W, device):
        if (D, H, W) != self.current_spatial_shape or self.hilbert_idx.device != device:
            h, inv_h = self._generate_3d_hilbert_curve(D, H, W, device)
            self.hilbert_idx = h
            self.inverse_hilbert_idx = inv_h
            self.current_spatial_shape = (D, H, W)

    def forward(self, x_list, mask):
        # **x_list contains four modality features, each with shape (B,C,D,H,W) / x_list 包含四个模态特征，每个形状为 (B,C,D,H,W)**
        # **mask has shape (B,4), where each entry marks whether the modality is present / mask 形状为 (B,4)，表示各模态是否存在**
        x = torch.stack(x_list, dim=1) 
        B, M, C, D, H, W = x.shape

        # **Prepare broadcast masks for 3D features and token sequences / 为 3D 特征和 token 序列准备广播 mask**
        mask_m_6d = mask.to(dtype=x.dtype).view(B, M, 1, 1, 1, 1)   # (B,M,1,1,1,1)

        # **Local per-modality convolution before token interaction / token 交互前的逐模态局部卷积**
        x_reshaped = rearrange(x, 'b m c d h w -> (b m) c d h w')
        x_conv = self.short_conv(x_reshaped)

        # **Flatten each modality feature map to tokens: (B,M,L,C) / 将每个模态特征图展平成 token: (B,M,L,C)**
        x = rearrange(x_conv, '(b m) c d h w -> b m c (d h w)', m=M).permute(0, 1, 3, 2)

        # **Re-mask after convolution because normalization/activation can turn zeroed inputs nonzero / 卷积后的归一化和激活可能使缺失模态非零，因此重新 mask**
        mask_m_4d = mask.to(dtype=x.dtype).view(B, M, 1, 1)         # (B,M,1,1)
        x = x * mask_m_4d

        # **Add modality embeddings only to present modalities / 仅对存在模态加入模态嵌入**
        x = x + self.modality_embed * mask_m_4d

        # **Keep missing-modality tokens as no-op after embedding / 加入嵌入后仍保持缺失模态为 no-op**
        x = x * mask_m_4d

        x = self.kda_norm(x)

        # **LayerNorm may introduce bias on zero tokens, so mask again / LayerNorm 可能使零 token 产生偏置，因此再次 mask**
        x = x * mask_m_4d

        # **Hilbert sorting preserves spatial locality before sequence modeling / Hilbert 排序在序列建模前保留空间局部性**
        self._update_hilbert_cache(D, H, W, x.device)
        x = x.index_select(2, self.hilbert_idx)   # dim=2 对 L 维重排


        # **Optional valid-modality sorting is available but disabled in the released HKFuse path / 可选 valid-modality 排序保留在此处，但当前 HKFuse 主路径未开启**
        sort_idx = None
        if self.enable_valid_sort:
            sort_idx = torch.argsort(mask.float(), dim=1, descending=True)
            # Batch-wise modality permutation without expanding indices to (B,M,L,C).
            batch = torch.arange(B, device=x.device)[:, None]
            x = x[batch, sort_idx]

            # **When enabled, the mask order must follow the same modality permutation / 启用时 mask 必须同步执行相同的模态重排**
            mask_sorted = mask[batch, sort_idx]
            mask_m_4d_sorted = mask_sorted.to(dtype=x.dtype).view(B, M, 1, 1)
            x = x * mask_m_4d_sorted
        else:
            mask_sorted = None

        # **Voxel-wise modality interleaving followed by KDA interaction / 按体素交错排列模态 token 后执行 KDA 交互**
        if self.training:
            x_fused_flat = checkpoint(self.kda, rearrange(x, 'b m l c -> b (l m) c'), use_reentrant=False)
        else:
            x_fused_flat = self.kda(rearrange(x, 'b m l c -> b (l m) c'))

        # **Restore modality slots after KDA / KDA 后恢复模态槽位**
        x_fused = rearrange(x_fused_flat, 'b (l m) c -> b m l c', m=M)
        if self.enable_valid_sort:
            # Inverse permutation back to original modality slots.
            batch = torch.arange(B, device=x_fused.device)[:, None]
            inv_sort_idx = torch.argsort(sort_idx, dim=1)
            x_restored = x_fused[batch, inv_sort_idx]
        else:
            x_restored = x_fused

        # **Mask KDA outputs so missing modalities cannot affect later modality averaging / 对 KDA 输出重新 mask，避免缺失模态影响后续模态平均**
        x_restored = x_restored * mask_m_4d

        # **Restore original spatial order / 恢复原始空间顺序**
        x_final = x_restored.index_select(2, self.inverse_hilbert_idx)

        # **Reshape back to 3D features and apply the final modality mask / 还原为 3D 特征并应用最终模态 mask**
        x_final = rearrange(x_final, 'b m (d h w) c -> b m c d h w', d=D, h=H, w=W)
        x_final = x_final * mask_m_6d

        return [x_final[:, i] for i in range(M)]

class TransformerAttention(nn.Module):
    """**Token self-attention used by intra-transformers and the bottleneck inter-transformer / 用于 intra-transformer 和瓶颈 inter-transformer 的 token self-attention**"""
    def __init__(self, dim, num_heads=8, rope3d: nn.Module | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # **RoPE is optional; HKFuse enables it only in the bottleneck inter-transformer / RoPE 是可选项，HKFuse 仅在瓶颈 inter-transformer 中启用**
        self.rope3d = rope3d

    def forward(self, x):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(self.norm(x))
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        # **Apply 3D RoPE to Q/K when provided / 提供 3D RoPE 时将其应用到 Q/K**
        if self.rope3d is not None:
            q, k = self.rope3d(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
