import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm
from torch.cuda.amp import autocast
from utils.initialization import InitWeights_He
from hkfuse_modules import BiDirectionalChunkKDA, HKBlock, RotaryPositionalEmbedding3D, TransformerAttention

basic_dims = 16  # **Base channel width / 基础通道宽度**
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
input_patch_size = 128



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))  # (B, 16, 128, 128, 128)

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))  # (B, 32, 64, 64, 64)

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))  # (B, 64, 32, 32, 32)

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))  # (B, 128, 16, 16, 16)

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))  # (B, 256, 8, 8, 8)

        return x1, x2, x3, x4, x5

# **Key-voxel router conditioned on the available-modality mask / 基于可用模态 mask 的关键体素路由头**
class KeyVoxelRouter(nn.Module):
    def __init__(self, in_channels, mask_channels=4):
        super().__init__()
        # **Embed the modality mask before voxel scoring / 将模态 mask 映射为嵌入后参与体素评分**
        self.mask_embed = nn.Sequential(
            nn.Linear(4, mask_channels),
            nn.SiLU()
        )
        # **Lightweight 1x1x1 scoring head / 轻量 1x1x1 评分头**
        self.net = nn.Sequential(
            nn.Conv3d(in_channels + mask_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, kernel_size=1, bias=True)
            # **Return logits for stable BCE training / 输出 logits 以便稳定计算 BCE loss**
        )

    def forward(self, base, mask):
        B, C, D, H, W = base.shape
        # **Broadcast mask embedding to the feature grid / 将 mask 嵌入广播到特征网格**
        m_emb = self.mask_embed(mask.float()) 
        m_emb = m_emb.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)
        # **Predict one key-voxel logit per spatial location / 为每个空间位置预测关键体素 logit**
        x = torch.cat([base, m_emb], dim=1) 
        logits = self.net(x) 
        return logits
    
class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        # **Align decoder features by padding when interpolation produces size offsets / 插值尺寸偏差时用 padding 对齐特征**
        def pad_to_match(src, target):
            dz = target.size(2) - src.size(2)
            dy = target.size(3) - src.size(3)
            dx = target.size(4) - src.size(4)
            if dz != 0 or dy != 0 or dx != 0:
                # **F.pad order is (W_left, W_right, H_left, H_right, D_left, D_right) / F.pad 顺序为 W、H、D 的前后补边**
                src = F.pad(src, (0, dx, 0, dy, 0, dz))
            return src
        
        de_x5 = self.d4_c1(self.d4(x5)) 
        de_x5 = pad_to_match(de_x5, x4)
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))

        de_x4 = self.d3_c1(self.d3(de_x4)) 
        de_x4 = pad_to_match(de_x4, x3)
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))

        de_x3 = self.d2_c1(self.d2(de_x3)) 
        de_x3 = pad_to_match(de_x3, x2)
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        
        de_x2 = self.d1_c1(self.d1(de_x2))
        de_x2 = pad_to_match(de_x2, x1)
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1)) 

        logits = self.seg_layer(de_x1)
        pred = logits

        return pred 


# ==========================
# HKFuse decoder
# ==========================

class Decoder_fuse_v3(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse_v3, self).__init__()
        
        # **Project bottleneck features back to the encoder channel width / 将瓶颈特征投影回编码器通道宽度**
        self.proj_x5 = nn.Conv3d(transformer_basic_dims, basic_dims*16, 1)

        # **Deep-supervision heads / 深度监督预测头**
        self.seg_d4 = nn.Conv3d(basic_dims*16, num_cls, 1)  # x5 -> Pred4
        self.seg_d3 = nn.Conv3d(basic_dims*8, num_cls, 1)  # d4 -> Pred3
        self.seg_d2 = nn.Conv3d(basic_dims*4, num_cls, 1)  # d3 -> Pred2
        self.seg_d1 = nn.Conv3d(basic_dims*2, num_cls, 1)  # d2 -> Pred1
        self.seg_final = nn.Conv3d(basic_dims, num_cls, 1)  # Final Pred

        # **Upsampling decoder blocks / 解码器上采样模块**
        # Stage 4: 256 + 128 -> 128
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16 + basic_dims*8, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        # Stage 3: 128 + 64 -> 64
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8 + basic_dims*4, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        # Stage 2: 64 + 32 -> 32
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4 + basic_dims*2, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        # Stage 1: 32 + 16 -> 16
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2 + basic_dims, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        # **Auxiliary predictions are upsampled to the input resolution / 辅助预测上采样到输入分辨率**
        self.up_scale2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up_scale4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up_scale8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up_scale16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        # **Pad decoder features to match skip-feature sizes / 对齐解码特征和 skip 特征尺寸**
        def pad_to_match(src, target):
            dz = target.size(2) - src.size(2)
            dy = target.size(3) - src.size(3)
            dx = target.size(4) - src.size(4)
            if dz != 0 or dy != 0 or dx != 0:
                src = F.pad(src, (0, dx, 0, dy, 0, dz))
            return src

        # **Reduce bottleneck channels before decoding: 512 -> 256 / 解码前先将瓶颈通道从 512 降到 256**
        x5 = self.proj_x5(x5)
        
        # Level 4
        pred4 = self.seg_d4(x5)
        up_x5 = self.up4(x5)
        up_x5 = pad_to_match(up_x5, x4)
        d4 = self.d4_out(self.d4_c2(torch.cat((x4, up_x5), dim=1)))
        
        # Level 3
        pred3 = self.seg_d3(d4)
        up_d4 = self.up3(d4)
        up_d4 = pad_to_match(up_d4, x3)
        d3 = self.d3_out(self.d3_c2(torch.cat((x3, up_d4), dim=1)))
        
        # Level 2
        pred2 = self.seg_d2(d3)
        up_d3 = self.up2(d3)
        up_d3 = pad_to_match(up_d3, x2)
        d2 = self.d2_out(self.d2_c2(torch.cat((x2, up_d3), dim=1)))
        
        # Level 1
        pred1 = self.seg_d1(d2)
        up_d2 = self.up1(d2)
        up_d2 = pad_to_match(up_d2, x1)
        d1 = self.d1_out(self.d1_c2(torch.cat((x1, up_d2), dim=1)))
        
        # Final
        # final_pred = self.softmax(self.seg_final(d1))
        final_pred = self.seg_final(d1)
        # **Return multi-scale auxiliary predictions / 返回多尺度辅助预测**
        aux_preds = (self.up_scale2(pred1), self.up_scale4(pred2), self.up_scale8(pred3), self.up_scale16(pred4))
        
        return final_pred, aux_preds

class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x
    

                                          
class HKFuse(nn.Module):
    def __init__(self, num_cls=4, interleaved_tokenization=False, kimi_skip=False):
        super(HKFuse, self).__init__()

        # **Modality-specific encoders / 各模态独立编码器**
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()


        # **Project modality bottleneck features to token channels / 将各模态瓶颈特征投影到 token 通道**
        self.flair_encode_proj = nn.Conv3d(basic_dims*16, transformer_basic_dims, 1)
        self.t1ce_encode_proj = nn.Conv3d(basic_dims*16, transformer_basic_dims, 1)
        self.t1_encode_proj = nn.Conv3d(basic_dims*16, transformer_basic_dims, 1)
        self.t2_encode_proj = nn.Conv3d(basic_dims*16, transformer_basic_dims, 1)

        # **Intra-modal transformer attention; RoPE is disabled before cross-modal fusion / 模态内 transformer attention，跨模态融合前不使用 RoPE**
        self.flair_intra_transformer = TransformerAttention(dim=transformer_basic_dims, num_heads=num_heads, rope3d=None)
        self.t1ce_intra_transformer  = TransformerAttention(dim=transformer_basic_dims, num_heads=num_heads, rope3d=None)
        self.t1_intra_transformer    = TransformerAttention(dim=transformer_basic_dims, num_heads=num_heads, rope3d=None)
        self.t2_intra_transformer    = TransformerAttention(dim=transformer_basic_dims, num_heads=num_heads, rope3d=None)


        # **Bottleneck HK-Block fusion with fixed modality slots and Hilbert ordering / 瓶颈 HK-Block 融合，固定模态槽位并使用 Hilbert 排序**
        self.bottleneck_hk_block = HKBlock(
            dim=transformer_basic_dims,
            spatial_size=(8, 8, 8),
            enable_valid_sort=False,
        )   

        # **3D RoPE provides spatial orientation for Hilbert-ordered bottleneck tokens / 3D RoPE 为 Hilbert 排序后的瓶颈 token 提供空间方向信息**
        post_spatial = (8, 8, 8)
        head_dim = transformer_basic_dims // num_heads  # 512//8=64

        self.inter_rope3d = RotaryPositionalEmbedding3D(
            spatial_shape=post_spatial,
            head_dim=head_dim,
        )

        self.inter_transformer = TransformerAttention(
            dim=transformer_basic_dims,
            num_heads=num_heads,
            rope3d=self.inter_rope3d,
        )


        self.masker = MaskModal()

        # **Optional skip-level HK-Block fusion / 可选的 skip 层 HK-Block 融合**
        self.kimi_skip = kimi_skip
        self.skip_config = [True, True, True, True] 
        
        if self.kimi_skip:
            self.skip_hk_blocks = nn.ModuleList([
                # **Each skip HK-Block uses the spatial size of its feature level / 每个 skip HK-Block 对应该层特征的空间尺寸**
                HKBlock(dim=basic_dims, spatial_size=(128, 128, 128), enable_valid_sort=False),     # 16
                HKBlock(dim=basic_dims*2, spatial_size=(64, 64, 64), enable_valid_sort=False),      # 32
                HKBlock(dim=basic_dims*4, spatial_size=(32, 32, 32), enable_valid_sort=False),      # 64
                HKBlock(dim=basic_dims*8, spatial_size=(16, 16, 16), enable_valid_sort=False)       # 128
            ])

        # **KVR scoring heads for skip-level key-voxel selection / 用于 skip 层关键体素选择的 KVR 评分头**
        self.key_voxel_routers = nn.ModuleList([
            KeyVoxelRouter(basic_dims),    # x1 (16通道)
            KeyVoxelRouter(basic_dims*2),  # x2 (32通道)
            KeyVoxelRouter(basic_dims*4),  # x3 (64通道)
            KeyVoxelRouter(basic_dims*8)   # x4 (128通道)
        ])

        # =========================
        # **KVR skip fusion configuration / KVR skip 融合配置**
        # **Only selected key voxels use cross-modal KDA, while the dense base uses lightweight convolution / 仅关键体素使用跨模态 KDA，dense base 使用轻量卷积**
        # =========================
        self.enable_kvr = True

        # **Curriculum schedule for KVR injection / KVR 注入强度的课程调度**
        # **Warmup phase: train the router while keeping KVR injection disabled / 预热阶段训练 router，但不注入 KVR 更新**
        self.kvr_gamma_max = 0.5
        self.kvr_warmup_epochs = 20 
        # **Injection phase: linearly increase the residual update strength / 注入阶段线性提高残差更新强度**
        self.kvr_gamma_start_epoch = 20    
        self.kvr_gamma_warmup_epochs = 20  
        self._train_epoch = 0  # **Updated by the training loop through set_train_epoch / 由训练循环通过 set_train_epoch 更新**


        # **Enable KVR independently for each skip level / 对每个 skip 层独立控制是否启用 KVR**
        # idx: 0->x1(128^3), 1->x2(64^3), 2->x3(32^3), 3->x4(16^3)
        self.kvr_skip_enable = [True, True, True, True]

        # **Top-k ratios keep each level close to K=200 selected voxels / top-k 比例使每层接近选择 K=200 个体素**
        self.key_voxel_ratio = [9.5367e-05, 7.6294e-04, 6.1035e-03, 4.8828e-02]

        # **Reserved weights for heuristic key-voxel scoring variants / 预留给启发式关键体素评分变体的权重**
        self.kvr_score_alpha = 0.7
        self.kvr_edge_weight = 0.3

        # **Lightweight local fusion for the dense skip base / dense skip base 的轻量局部融合**
        def _lite_block(ch):
            return nn.Sequential(
                nn.Conv3d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=True),
                nn.GroupNorm(max(1, ch // 8), ch),
                nn.SiLU(),
                nn.Conv3d(ch, ch, kernel_size=1, bias=True),
            )

        # **Smooth the sparse residual update after scattering back to the dense grid / sparse 残差写回 dense 网格后进行平滑**
        self.kvr_refine_enable = True

        self.skip_light_fuse = nn.ModuleList([
            _lite_block(basic_dims),       # x1: 16
            _lite_block(basic_dims * 2),   # x2: 32
            _lite_block(basic_dims * 4),   # x3: 64
            _lite_block(basic_dims * 8),   # x4: 128
        ])

        self.kvr_refine = nn.ModuleList([
            _lite_block(basic_dims),
            _lite_block(basic_dims * 2),
            _lite_block(basic_dims * 4),
            _lite_block(basic_dims * 8),
        ])



        # **Segmentation decoders / 分割解码器**
        self.decoder_fuse = Decoder_fuse_v3(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.apply(InitWeights_He(1e-2))
        
        # **Numerically stable KDA initialization / KDA 数值稳定初始化**
        if hasattr(self, '_fix_kda_initialization'):
            self._fix_kda_initialization()

    def set_train_epoch(self, epoch: int):
        self._train_epoch = int(epoch)

    # **KVR gamma schedule controls when sparse residual updates are injected / KVR gamma 调度控制 sparse 残差更新何时注入**
    def get_kvr_gamma(self) -> float:
        # **Router warmup keeps the backbone path unchanged / Router 预热阶段保持主干路径不变**
        if self._train_epoch < self.kvr_gamma_start_epoch:
            return 0.0
            
        # **Linear ramp avoids abrupt changes in the sparse update strength / 线性升温避免 sparse 更新强度突变**
        if self.kvr_gamma_warmup_epochs <= 0:
            return float(self.kvr_gamma_max)
            
        t = (self._train_epoch - self.kvr_gamma_start_epoch + 1) / float(self.kvr_gamma_warmup_epochs)
        t = max(0.0, min(1.0, t))
        return float(self.kvr_gamma_max) * t

    def freeze_dense_skip_hk_blocks(self):
        # **The sparse KVR path bypasses HKBlock.short_conv for enabled skip levels / 启用 KVR 的 skip 层会绕过 HKBlock.short_conv**
        if not hasattr(self, "skip_hk_blocks"):
            return
        for idx, enabled in enumerate(self.kvr_skip_enable):
            if not enabled or idx >= len(self.skip_hk_blocks):
                continue
            for p in self.skip_hk_blocks[idx].short_conv.parameters():
                p.requires_grad = False


    def _fix_kda_initialization(self):
        print(">>> [HKFuse] 正在执行 KDA 参数清洗 (Stable Mode)...")
        for name, module in self.named_modules():
            if isinstance(module, BiDirectionalChunkKDA):
                # **Initialize A_log in a stable negative range for long token sequences / 将 A_log 初始化到适合长序列的稳定负值范围**
                torch.nn.init.uniform_(module.A_log, -4, -2)
                
                # **Zero dt_bias to avoid a large initial gate offset / 将 dt_bias 置零以避免初始 gate 偏移过大**
                if hasattr(module, 'dt_bias'):
                    torch.nn.init.zeros_(module.dt_bias)
                    
                # **Use small gate-projection weights so A_log dominates early dynamics / 使用较小 gate 投影权重，使早期动态主要由 A_log 控制**
                torch.nn.init.normal_(module.f_a_proj.weight, std=0.01)
                torch.nn.init.normal_(module.f_b_proj.weight, std=0.01)
        print(">>> [HKFuse] KDA 参数重置完成。")

    def forward(self, x, mask):
        # **Modality-specific encoding / 各模态独立编码**
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        # **Intra-modal bottleneck projection and transformer attention / 模态内瓶颈投影与 transformer attention**
        flair_feat = self.flair_encode_proj(flair_x5) 
        t1ce_feat = self.t1ce_encode_proj(t1ce_x5)
        t1_feat = self.t1_encode_proj(t1_x5)
        t2_feat = self.t2_encode_proj(t2_x5)

        def apply_transformer_3d(feat_bcdhw, transformer_module):
            """
            feat_bcdhw: (B,C,D,H,W)
            return:     (B,C,D,H,W)
            **Flatten the 3D grid to token sequence for self-attention / 将 3D 网格展平成 token 序列以执行 self-attention**
            """
            B, C, D, H, W = feat_bcdhw.shape
            x_seq = feat_bcdhw.view(B, C, D*H*W).permute(0, 2, 1).contiguous()  # (B,N,C)
            x_seq = x_seq + transformer_module(x_seq)  # **Residual refinement / 残差式特征细化**
            out = x_seq.permute(0, 2, 1).contiguous().view(B, C, D, H, W)
            return out

        flair_intra = apply_transformer_3d(flair_feat, self.flair_intra_transformer)
        t1ce_intra  = apply_transformer_3d(t1ce_feat,  self.t1ce_intra_transformer)
        t1_intra    = apply_transformer_3d(t1_feat,    self.t1_intra_transformer)
        t2_intra    = apply_transformer_3d(t2_feat,    self.t2_intra_transformer)


        # **Single-modality deep supervision during training / 训练阶段的单模态深度监督**
        if self.training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        
        # **Apply the available-modality mask before skip fusion / skip 融合前应用可用模态 mask**
        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)

        # =========================
        # **Mask-aware modality mean avoids magnitude dilution from missing modalities / mask-aware 模态均值避免缺失模态的零值稀释幅值**
        # x_bmcdhw: (B, M=4, C, D, H, W)
        # mask: (B, 4) -> {0/1} 或 bool
        # =========================
        def masked_modal_mean(x_bmcdhw, mask_bm, eps: float = 1e-6):
            B, M = x_bmcdhw.shape[0], x_bmcdhw.shape[1]
            w = mask_bm.to(dtype=x_bmcdhw.dtype).view(B, M, 1, 1, 1, 1)  # (B,M,1,1,1,1)
            denom = w.sum(dim=1)  # (B,1,1,1,1)
            denom = torch.clamp(denom, min=eps)  # **Avoid division by zero / 避免除零**
            return (x_bmcdhw * w).sum(dim=1) / denom  # -> (B,C,D,H,W)

        # =========================
        # **KVR helper functions for scoring, top-k selection, and Hilbert sorting / KVR 辅助函数：评分、top-k 选点和 Hilbert 排序**
        # =========================
        def masked_modal_var(x_bmcdhw, mask_bm, eps: float = 1e-6):
            """
            x_bmcdhw: (B,4,C,D,H,W)
            mask_bm:  (B,4)
            return:   (B,D,H,W)  **Modality-disagreement score averaged over channels / 按通道平均后的跨模态分歧强度**
            **This helper is kept for heuristic scoring variants / 该辅助函数保留给启发式评分变体使用**
            """
            B, M = x_bmcdhw.shape[0], x_bmcdhw.shape[1]
            w = mask_bm.to(dtype=x_bmcdhw.dtype).view(B, M, 1, 1, 1, 1)
            denom = w.sum(dim=1).clamp_min(eps)  # (B,1,1,1,1)
            mean = (x_bmcdhw * w).sum(dim=1) / denom                       # (B,C,D,H,W)
            var = ((x_bmcdhw - mean.unsqueeze(1)) ** 2 * w).sum(dim=1) / denom
            return var.mean(dim=1)  # (B,D,H,W)

        def grad_mag_3d(x_bcdhw):
            """
            x_bcdhw: (B,C,D,H,W)
            return:  (B,D,H,W)  **3D edge-strength score averaged over channels / 按通道平均后的 3D 边界强度**
            """
            # z
            dz = (x_bcdhw[:, :, 1:, :, :] - x_bcdhw[:, :, :-1, :, :]).abs()
            dz = F.pad(dz, (0,0,0,0,1,0))
            # y
            dy = (x_bcdhw[:, :, :, 1:, :] - x_bcdhw[:, :, :, :-1, :]).abs()
            dy = F.pad(dy, (0,0,1,0,0,0))
            # x
            dx = (x_bcdhw[:, :, :, :, 1:] - x_bcdhw[:, :, :, :, :-1]).abs()
            dx = F.pad(dx, (1,0,0,0,0,0))
            g = (dx + dy + dz).mean(dim=1)  # (B,D,H,W)
            return g

        def select_topk_idx(score_bdhw, ratio: float):
            """
            score_bdhw: (B,D,H,W)
            return: topk_idx (B,K) in [0, L)
            """
            B, D, H, W = score_bdhw.shape
            L = D * H * W
            K = max(1, int(L * ratio))
            score_flat = score_bdhw.reshape(B, L).detach()  # **Top-k selection is non-differentiable / top-k 选择不参与反传**
            topk_idx = torch.topk(score_flat, k=K, dim=1, largest=True, sorted=False).indices
            return topk_idx  # (B,K)

        def hilbert_sort_topk(topk_idx_bk, fuser, D, H, W):
            """
            topk_idx_bk: (B,K)
            **Sort selected voxels by Hilbert rank before KDA / 在 KDA 前按 Hilbert 序重排选中的体素**
            """
            # **Ensure the HK-Block Hilbert cache matches the current feature size / 确保 HK-Block 的 Hilbert 缓存匹配当前特征尺寸**
            fuser._update_hilbert_cache(D, H, W, topk_idx_bk.device)
            rank = fuser.inverse_hilbert_idx  # (L,)  rank[pos] = Hilbert 序号
            ranks = rank[topk_idx_bk]         # (B,K)
            order = torch.argsort(ranks, dim=1)
            return torch.gather(topk_idx_bk, 1, order)

        # **Store router logits with their stage index for KVR auxiliary losses / 保存带 stage index 的 router logits，供 KVR 辅助 loss 使用**
        kvr_router_outputs = []

        if self.kimi_skip:
            def run_kvr_skip(feat, idx):
                """
                feat: (B, 4*C, D, H, W)
                return: (B, C, D, H, W)
                **Dense skip base plus KVR sparse residual update / dense skip base 加 KVR sparse 残差更新**
                """
                B, KC, D, H, W = feat.shape
                C = KC // 4
                feat_6d = feat.view(B, 4, C, D, H, W)  # (B,4,C,D,H,W)

                # **Build a dense base with mask-aware mean and local convolution / 用 mask-aware mean 和局部卷积构建 dense base**
                base = masked_modal_mean(feat_6d, mask)                 # (B,C,D,H,W)
                base = self.skip_light_fuse[idx](base)                  # (B,C,D,H,W)

                # **Router logits are predicted from a detached base to isolate auxiliary supervision / 从 detach 后的 base 预测 router logits，隔离辅助监督**
                router_in = base.detach()
                router_logits = self.key_voxel_routers[idx](router_in, mask)
                stage_idx = torch.tensor(idx, device=router_logits.device, dtype=torch.long)
                kvr_router_outputs.append((stage_idx, router_logits))

                # **Skip KVR selection when residual injection is inactive / 残差注入未启用时跳过 KVR 选点**
                gamma = self.get_kvr_gamma()
                if (gamma <= 0.0) or (self.key_voxel_ratio[idx] <= 0):
                    return base

                # **Use edge strength during warmup, then switch to learned router scores / 预热期使用边界强度，之后切换到学习到的 router score**
                if self._train_epoch < self.kvr_warmup_epochs:
                    score = grad_mag_3d(base)
                else:
                    score = router_logits.squeeze(1)


                # **Hard top-k key-voxel selection / hard top-k 关键体素选择**
                topk_idx = select_topk_idx(score, self.key_voxel_ratio[idx])
                
                # **Hilbert sorting -> KDA interaction -> scatter residual update / Hilbert 排序 -> KDA 交互 -> scatter 残差更新**
                topk_idx = hilbert_sort_topk(topk_idx, self.skip_hk_blocks[idx], D, H, W)

                # **Gather four-modality tokens at selected key voxels / 收集关键体素处的四模态 token**
                L = D * H * W
                feat_flat = feat_6d.view(B, 4, C, L).permute(0, 1, 3, 2)            # (B,4,L,C)
                idx_exp = topk_idx.view(B, 1, -1, 1).expand(B, 4, topk_idx.shape[1], C)
                tokens = torch.gather(feat_flat, 2, idx_exp)                        # (B,4,K,C)

                # **Keep missing modalities as no-op tokens through mask-gated embedding and normalization / 通过 mask-gated embedding 与归一化使缺失模态保持 no-op**
                mask_m = mask.to(dtype=tokens.dtype).view(B, 4, 1, 1)
                tokens = tokens * mask_m
                tokens = tokens + self.skip_hk_blocks[idx].modality_embed * mask_m
                tokens = tokens * mask_m
                tokens = self.skip_hk_blocks[idx].kda_norm(tokens)
                tokens = tokens * mask_m

                # **Interleave modality tokens per voxel before KDA / KDA 前按体素交错排列各模态 token**
                tokens_inter = tokens.permute(0, 2, 1, 3).reshape(B, -1, C)         # (B,K*4,C)
                tokens_out = self.skip_hk_blocks[idx].kda(tokens_inter)                # (B,K*4,C)
                tokens_out = tokens_out.view(B, -1, 4, C).permute(0, 2, 1, 3)       # (B,4,K,C)
                tokens_out = tokens_out * mask_m

                # **Aggregate present modalities only when forming fused key-voxel features / 聚合关键体素特征时只使用存在的模态**
                denom = mask_m.sum(dim=1).clamp_min(1.0)              # (B,1,1)
                denom = denom.expand(B, tokens_out.shape[2], C)       # (B,K,C)
                fused_k = (tokens_out * mask_m).sum(dim=1) / denom    # (B,K,C)

                # **Scatter sparse residual updates back to the dense feature grid / 将 sparse 残差更新写回 dense 特征网格**
                base_flat = base.view(B, C, L).permute(0, 2, 1).contiguous()        # (B,L,C)
                batch = torch.arange(B, device=base_flat.device)[:, None]
                base_sel = base_flat[batch, topk_idx]
                fused_k = fused_k.to(dtype=base_sel.dtype)
                base_flat[batch, topk_idx] = base_sel + gamma * (fused_k - base_sel)  # delta injection
                out = base_flat.permute(0, 2, 1).view(B, C, D, H, W)               # (B,C,D,H,W)

                # **Smooth sparse-updated regions with local convolution / 使用局部卷积平滑 sparse 更新区域**
                if self.kvr_refine_enable:
                    out = self.kvr_refine[idx](out)

                return out

            def run_skip(feat, idx, enable):
                # feat: (B, 4*C, D, H, W)
                B, KC, D, H, W = feat.shape
                C = KC // 4
                feat_6d = feat.view(B, 4, C, D, H, W)

                # **Fallback skip path: mask-aware mean followed by local convolution / fallback skip 路径：mask-aware mean 后接局部卷积**
                if not enable:
                    base = masked_modal_mean(feat_6d, mask)   # (B,C,D,H,W)
                    base = self.skip_light_fuse[idx](base)    # (B,C,D,H,W)
                    return base


                # **Use KVR when enabled for this skip level; otherwise use dense HK-Block fusion / 该 skip 层启用 KVR 时走 KVR，否则走 dense HK-Block 融合**
                if self.enable_kvr and self.kvr_skip_enable[idx]:
                    return run_kvr_skip(feat, idx)

                # **Dense HK-Block skip fusion / dense HK-Block skip 融合**
                inputs = [feat_6d[:, i] for i in range(4)]
                fused_list = self.skip_hk_blocks[idx](inputs, mask)                    # List[4]
                fused_stack = torch.stack(fused_list, dim=1)                        # (B,4,C,D,H,W)
                return masked_modal_mean(fused_stack, mask)

            x1 = run_skip(x1, 0, self.skip_config[0])
            x2 = run_skip(x2, 1, self.skip_config[1])
            x3 = run_skip(x3, 2, self.skip_config[2])
            x4 = run_skip(x4, 3, self.skip_config[3])


        else:
            # **Without skip HK-Blocks, use mask-aware mean on reshaped modality features / 不使用 skip HK-Block 时，对 reshape 后的模态特征做 mask-aware mean**
            def simple_fuse(f):
                B, KC, D, H, W = f.shape
                C = KC // 4
                f = f.view(B, 4, C, D, H, W)
                return masked_modal_mean(f, mask)

            x1, x2, x3, x4 = [simple_fuse(f) for f in [x1, x2, x3, x4]]


        # **Bottleneck HK-Block fusion over masked modality tokens / 对 mask 后的模态 token 执行瓶颈 HK-Block 融合**
        intra_stack = torch.stack((flair_intra, t1ce_intra, t1_intra, t2_intra), dim=1)
        intra_masked_flat = self.masker(intra_stack, mask)
        intra_feats = torch.split(intra_masked_flat, transformer_basic_dims, dim=1)
        fused_list = self.bottleneck_hk_block(intra_feats, mask)

        # **Fuse only present modality outputs from the bottleneck HK-Block / 只融合瓶颈 HK-Block 中存在模态的输出**
        fused_stack = torch.stack(fused_list, dim=1)  # (B,4,512,8,8,8)
        x5_inter = masked_modal_mean(fused_stack, mask)

        # **Inter-transformer refinement with RoPE-guided spatial orientation for Hilbert-ordered bottleneck tokens / inter-transformer 细化，并用 RoPE 为 Hilbert 排序后的瓶颈 token 提供空间方向信息**
        x5_inter = apply_transformer_3d(x5_inter, self.inter_transformer)


        # **The decoder projects the 512-channel fused bottleneck back to encoder channels / 解码器会将 512 通道融合瓶颈投影回编码器通道**
        fuse_pred, aux_preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        
        if self.training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), aux_preds, kvr_router_outputs
        else:
            return fuse_pred
