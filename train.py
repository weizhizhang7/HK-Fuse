#coding=utf-8
import argparse
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.optim
import sys
import types
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from utils.random_seed import setup_seed
from HKFuse import HKFuse as Model
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from utils import Parser,criterions2
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from predict2 import AverageMeter, test_softmax

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0.05, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--kimi_skip', action='store_true', default=False)
parser.add_argument('--tc_dice_weight', default=1.0, type=float, help='Class weight for TC-related Dice terms.')
parser.add_argument('--kvr_gamma_max', default=0.5, type=float)
parser.add_argument('--disable_kvr', action='store_true', default=False)
parser.add_argument('--freeze_dead_params', action='store_true', default=False)
parser.add_argument('--kvr_warmup_epochs', default=20, type=int, help='Epochs before KVR residual updates are injected.')
parser.add_argument('--kvr_gamma_start_epoch', default=20, type=int, help='Epoch to start KVR residual injection.')
parser.add_argument('--kvr_gamma_warmup_epochs', default=20, type=int, help='Epochs used to ramp KVR gamma to its maximum.')
parser.add_argument('--lambda_kvr_router', default=1.0, type=float, help='Weight for KVR router boundary supervision.')
parser.add_argument('--lambda_kvr_budget', default=0.05, type=float, help='Weight for KVR key-voxel budget regularization.')
# **Numerical debugging controls are disabled by default for long training runs / 数值调试开关默认关闭，避免影响长时间训练**
parser.add_argument('--kda_debug', action='store_true', default=False, help='Patch KDA core to exit on non-finite q/k/v/g/beta or output.')
parser.add_argument('--nan_debug', action='store_true', default=False, help='Enable module output hooks and grad/param checks.')
parser.add_argument('--nan_debug_start_step', default=180, type=int)
parser.add_argument('--nan_debug_exit', action='store_true', default=False, help='Exit immediately when forward-hook sees non-finite output.')
parser.add_argument('--nan_grad_action', default='skip_step', type=str, choices=['skip_step', 'exit'])
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline', 'disabled'])
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
# writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

val_check = [1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 950, 970, 980, 990, 995, 1000] 
print(f"Validation checks: {val_check}")

def main():
    ##########setting seed
    setup_seed(args.seed)
    
    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID") 
    wandb_name_and_id = f'BraTS23_HKFuse_K200_{"Skip" if args.kimi_skip else "NoSkip"}_epoch{args.num_epochs}_jobid{slurm_job_id}'    
    
    wandb_mode = args.wandb_mode
    wandb.init(
        project="HK-Fuse",
        name=wandb_name_and_id,
        # entity="NeuroTumor",
        id=wandb_name_and_id,
        mode=wandb_mode,
        resume="allow",
        config={
            "architecture": "HKFuse",
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "iter_per_epoch": args.iter_per_epoch,
            "num_epochs": args.num_epochs,
            "datapath": args.datapath,
            "region_fusion_start_epoch": args.region_fusion_start_epoch,
            "interleaved_tokenization": args.interleaved_tokenization,
            "kimi_skip": args.kimi_skip
        }
    )
    
    ##########setting models
    if args.dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = Model(
                    num_cls=num_cls, 
                    interleaved_tokenization=args.interleaved_tokenization,
                    kimi_skip=args.kimi_skip,
            )
    print (model)
    # =========================
    # **RoPE runtime check before DataParallel wrapping / DataParallel 包装前执行 RoPE 运行检查**
    # =========================
    if hasattr(model, "inter_transformer"):
        rope_obj = getattr(model.inter_transformer, "rope3d", None)
        print(f"[RoPE Check] inter_transformer.rope3d is None? {rope_obj is None}", flush=True)
        if rope_obj is not None:
            # **The released bottleneck uses 8^3 tokens and one RoPE split per attention head / 当前瓶颈层使用 8^3 token，并为每个 attention head 配置 RoPE 维度切分**
            print(f"[RoPE Check] spatial_shape={getattr(rope_obj, 'spatial_shape', None)}, "
                f"head_dim={getattr(rope_obj, 'head_dim', None)}, "
                f"dz/dy/dx={getattr(rope_obj, 'dz', None)}/{getattr(rope_obj, 'dy', None)}/{getattr(rope_obj, 'dx', None)}",
                flush=True)
    else:
        print("[RoPE Check] model has no attribute inter_transformer (maybe wrong Model import?)", flush=True)
    model = torch.nn.DataParallel(model).cuda()

    # **Wire KVR schedule arguments into the underlying model module / 将 KVR 调度参数写入 DataParallel 内部模型**
    base_model = model.module if hasattr(model, "module") else model
    if hasattr(base_model, "kvr_gamma_max"):
        base_model.kvr_gamma_max = float(args.kvr_gamma_max)
    if args.disable_kvr and hasattr(base_model, "enable_kvr"):
        base_model.enable_kvr = False
    if args.freeze_dead_params and hasattr(base_model, "freeze_dense_skip_hk_blocks"):
        base_model.freeze_dense_skip_hk_blocks()
    if hasattr(base_model, "kvr_warmup_epochs"):
        base_model.kvr_warmup_epochs = args.kvr_warmup_epochs
        base_model.kvr_gamma_start_epoch = args.kvr_gamma_start_epoch
        base_model.kvr_gamma_warmup_epochs = args.kvr_gamma_warmup_epochs
        logging.info(f"[KVR] warmup_epochs={args.kvr_warmup_epochs}, gamma_start_epoch={args.kvr_gamma_start_epoch}")

    def _tstat(name, t):
        t = t.detach()
        finite = torch.isfinite(t)
        if finite.any():
            tt = t[finite]
            return f"{name}: dtype={t.dtype} shape={tuple(t.shape)} min={tt.min().item():.3e} max={tt.max().item():.3e} mean={tt.mean().item():.3e}"
        return f"{name}: dtype={t.dtype} shape={tuple(t.shape)} ALL_NONFINITE"

    def patch_kda_debug(base_model):
        # **Optional KDA-core debug patch: fail fast when non-finite values appear / 可选 KDA 核心调试补丁：出现非有限值时立即退出**
        for mod_name, m in base_model.named_modules():
            if m.__class__.__name__ != "BiDirectionalChunkKDA":
                continue

            orig = m.forward_kda_core
            tag = mod_name
            def new_forward_kda_core(self, q, k, v, g, beta, _tag=tag, _orig=orig):
                
                if (not torch.isfinite(q).all()) or (not torch.isfinite(k).all()) or (not torch.isfinite(v).all()) or (not torch.isfinite(g).all()) or (not torch.isfinite(beta).all()):
                    print(f"\n[NONFINITE INPUT] {_tag}.forward_kda_core")
                    print(_tstat("q", q)); print(_tstat("k", k)); print(_tstat("v", v))
                    print(_tstat("g", g)); print(_tstat("beta", beta))
                    raise SystemExit(21)

                out = _orig(q, k, v, g, beta)

                if not torch.isfinite(out).all():
                    print(f"\n[NONFINITE OUTPUT] {_tag}.forward_kda_core")
                    print(_tstat("q", q)); print(_tstat("k", k)); print(_tstat("v", v))
                    print(_tstat("g", g)); print(_tstat("beta", beta))
                    print(_tstat("out", out))
                    raise SystemExit(22)

                return out

            m.forward_kda_core = types.MethodType(new_forward_kda_core, m)

    if args.kda_debug:
        patch_kda_debug(base_model)

    def _init_hilbert_buffers(model):
        device = next(model.parameters()).device
        try:
            if hasattr(model, "bottleneck_hk_block"):
                f = model.bottleneck_hk_block
                if hasattr(f, "_update_hilbert_cache"):
                    f._update_hilbert_cache(8, 8, 8, device)
            if hasattr(model, "skip_hk_blocks"):
                sizes = [(128, 128, 128), (64, 64, 64), (32, 32, 32), (16, 16, 16)]
                for f, s in zip(model.skip_hk_blocks, sizes):
                    if hasattr(f, "_update_hilbert_cache"):
                        f._update_hilbert_cache(s[0], s[1], s[2], device)
        except Exception as e:
            logging.warning(f"[Hilbert] init hilbert buffers failed: {e}")


    # ===================== Numerical Debug Hooks =====================
    NAN_DEBUG = bool(args.nan_debug)
    DEBUG_START_STEP = int(args.nan_debug_start_step)
    # **Gradient action for non-finite values: skip the step or stop immediately / 非有限梯度处理方式：跳过当前 step 或立即退出**
    NAN_GRAD_FUSE_ACTION = args.nan_grad_action

    def _isfinite_all(x: torch.Tensor) -> bool:
        return torch.isfinite(x).all().item()

    def _walk_any_nonfinite(obj) -> bool:
        if torch.is_tensor(obj):
            return (not _isfinite_all(obj))
        if isinstance(obj, (list, tuple)):
            return any(_walk_any_nonfinite(x) for x in obj)
        if isinstance(obj, dict):
            return any(_walk_any_nonfinite(v) for v in obj.values())
        return False

    def _first_nonfinite_path(tag, obj):
        if torch.is_tensor(obj):
            if not _isfinite_all(obj):
                bad = (~torch.isfinite(obj)).sum().item()
                print(f"\n[NaN/Inf DETECTED] {tag} bad={bad} shape={tuple(obj.shape)} dtype={obj.dtype}")
                return True
            return False
        if isinstance(obj, (list, tuple)):
            for i, it in enumerate(obj):
                if _first_nonfinite_path(f"{tag}[{i}]", it):
                    return True
            return False
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _first_nonfinite_path(f"{tag}.{k}", v):
                    return True
            return False
        return False

    _nan_debug_active = {"on": False}

    def make_forward_hook(name):
        def hook(mod, inp, out):
            if not _nan_debug_active["on"]:
                return
            if _walk_any_nonfinite(out):
                _first_nonfinite_path(f"{name}.out", out)
                if args.nan_debug_exit:
                    raise SystemExit(10)
                return
        return hook

    def check_grads_and_params(base_model):
        if not _nan_debug_active["on"]:
            return True
        for n, p in base_model.named_parameters():
            if p.grad is not None and (not _isfinite_all(p.grad)):
                bad = (~torch.isfinite(p.grad)).sum().item()
                print(f"\n[NaN/Inf GRAD] {n} bad={bad} shape={tuple(p.grad.shape)} dtype={p.grad.dtype}")
                if NAN_GRAD_FUSE_ACTION == "exit":
                    raise SystemExit(11)
                return False  
            if not _isfinite_all(p):
                bad = (~torch.isfinite(p)).sum().item()
                print(f"\n[NaN/Inf PARAM] {n} bad={bad} shape={tuple(p.shape)} dtype={p.dtype}")
                raise SystemExit(12)
        return True

    if NAN_DEBUG:
        for name, m in base_model.named_modules():
            cls = m.__class__.__name__
            if cls in {"HKBlock", "BiDirectionalChunkKDA", "TransformerAttention"}:
                m.register_forward_hook(make_forward_hook(f"{name}<{cls}>"))
    # ==============================================================

    logging.info(f"[DEBUG FLAGS] NAN_DEBUG={NAN_DEBUG} DEBUG_START_STEP={DEBUG_START_STEP} kda_debug={args.kda_debug}")
    logging.info(f"[DEBUG ARGS] kimi_skip={args.kimi_skip} batch_size={args.batch_size} lr={args.lr} weight_decay={args.weight_decay} num_epochs={args.num_epochs}")
    print(f"[DEBUG FLAGS] NAN_DEBUG={NAN_DEBUG} DEBUG_START_STEP={DEBUG_START_STEP}", flush=True)




    ########## Setting learning scheduler and optimizer
    # lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.RAdam(train_params)
    lr_schedule = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=args.num_epochs,
                eta_min=5e-6,
                last_epoch=-1
                )

    ########## Setting data
    if args.dataname in ['BRATS2023', 'BRATS2020', 'BRATS2015']:
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits.csv'
        val_file = 'datalist/val15splits.csv'

    elif args.dataname == 'BRATS2018':
        #### BRATS2018 contains three splits (1,2,3)
        test_file = 'datalist/Brats18_test15splits.csv'
        val_file = 'datalist/Brats18_val15splits.csv'
        train_file = 'datalist/train3.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, 
                                    root=args.datapath, 
                                    num_cls=num_cls, 
                                    train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, 
                                    root=args.datapath, 
                                    test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, 
                                    root=args.datapath, 
                                    num_cls=num_cls, 
                                    val_file=val_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch # iter=训练命令/默认
    iter_per_epoch = len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0

    ##########Resume Training
    if args.resume is not None:
        _init_hilbert_buffers(base_model)
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        # Restore LR scheduler state for consistent LR after resume.
        if 'lr_scheduler' in checkpoint:
            lr_schedule.load_state_dict(checkpoint['lr_scheduler'])
        else:
            # Backward-compatible fallback: align scheduler epoch counter with checkpoint.
            lr_schedule.last_epoch = int(checkpoint.get('epoch', -1))
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.num_epochs):
        # step_lr = lr_schedule(optimizer, epoch)
        # writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        model.train()
        model.module.is_training = True
        if hasattr(base_model, "set_train_epoch"):
            base_model.set_train_epoch(epoch)

        prm_cross_loss_epoch = 0.0
        prm_dice_loss_epoch = 0.0
        fuse_cross_loss_epoch = 0.0
        fuse_dice_loss_epoch = 0.0
        sep_cross_loss_epoch = 0.0
        sep_dice_loss_epoch = 0.0
        # **Track KVR auxiliary losses at epoch level / 按 epoch 统计 KVR 辅助 loss**
        kvr_router_loss_epoch = 0.0
        kvr_budget_loss_epoch = 0.0
        loss_epoch = 0.0

        ########## training epoch
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch

            _nan_debug_active["on"] = (NAN_DEBUG and step >= DEBUG_START_STEP)

            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]  # x=(B, M=4, 128, 128, 128), target=(B, C, 128, 128, 128), mask=(B, 4)
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            with torch.cuda.amp.autocast( dtype=torch.bfloat16):
                fuse_pred, sep_preds, prm_preds, kvr_router_outputs = model(x, mask)

            # **TC-weighted Dice can emphasize tumor-core classes during training / TC 加权 Dice 可在训练中增强 tumor-core 类别权重**
            def dice_loss_tc_weighted(prob, target_onehot, num_cls: int, tc_weight: float, eps: float = 1e-7):
                target_f = target_onehot.float()
                weights = torch.ones(num_cls, device=prob.device, dtype=prob.dtype)
                if num_cls >= 4 and tc_weight != 1.0:
                    weights[1] = tc_weight
                    weights[3] = tc_weight
                dice_terms = []
                for cls_idx in range(num_cls):
                    num = torch.sum(prob[:, cls_idx] * target_f[:, cls_idx])
                    den = torch.sum(prob[:, cls_idx]) + torch.sum(target_f[:, cls_idx]) + eps
                    dice_terms.append(2.0 * num / den)
                dice_terms = torch.stack(dice_terms)
                return 1.0 - (dice_terms * weights).sum() / weights.sum()

            ###Loss compute
            # **Fuse branch loss / 融合分支损失**
            fuse_cross_loss = criterions2.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)


            fuse_pred_prob = torch.softmax(fuse_pred, dim=1)
            fuse_dice_loss = dice_loss_tc_weighted(fuse_pred_prob, target, num_cls=num_cls, tc_weight=args.tc_dice_weight)
            
            fuse_loss = fuse_cross_loss + fuse_dice_loss
            fuse_cross_loss_epoch += fuse_cross_loss.item()
            fuse_dice_loss_epoch += fuse_dice_loss.item()


            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            
            # **Single-modality auxiliary losses / 单模态辅助损失**
            for sep_pred in sep_preds:
                
                sep_cross_loss += criterions2.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                
                
                sep_pred_prob = torch.softmax(sep_pred, dim=1)
                sep_dice_loss += dice_loss_tc_weighted(sep_pred_prob, target, num_cls=num_cls, tc_weight=args.tc_dice_weight)
                
            sep_loss = sep_cross_loss + sep_dice_loss
            sep_cross_loss_epoch += sep_cross_loss.item()
            sep_dice_loss_epoch += sep_dice_loss.item()

           
            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            
            # **Pyramid auxiliary losses / 金字塔辅助损失**
            for prm_pred in prm_preds:
                
                prm_cross_loss += criterions2.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                
                
                prm_pred_prob = torch.softmax(prm_pred, dim=1)
                prm_dice_loss += dice_loss_tc_weighted(prm_pred_prob, target, num_cls=num_cls, tc_weight=args.tc_dice_weight)
                
            prm_loss = prm_cross_loss + prm_dice_loss
            prm_cross_loss_epoch += prm_cross_loss.item()
            prm_dice_loss_epoch += prm_dice_loss.item()

            loss_kvr_router = torch.zeros(1).cuda()
            loss_kvr_budget = torch.zeros(1).cuda()


            if len(kvr_router_outputs) > 0:
                with torch.no_grad():
                    # **Use the tumor-core boundary as auxiliary supervision for KVR scoring / 使用 tumor-core 边界作为 KVR 评分的辅助监督**
                    roi_mask = (target[:, 1] + target[:, 3]) > 0.5
                    roi_mask = roi_mask.float().unsqueeze(1)  # (B, 1, D, H, W)

                    # **Boundary band from dilation minus erosion / 由膨胀减腐蚀得到边界带**
                    dilated = F.max_pool3d(roi_mask, kernel_size=5, stride=1, padding=2)
                    eroded = -F.max_pool3d(-roi_mask, kernel_size=5, stride=1, padding=2)
                    boundary_target = dilated - eroded

                bce_pos_weight = torch.tensor([20.0], device=x.device)  # **Positive weight for sparse boundary voxels / 稀疏边界体素的正样本权重**

                for stage_idx, logits in kvr_router_outputs:
                    # **Match the boundary target to each KVR skip level / 将边界监督下采样到对应 KVR skip 层**
                    curr_target = F.interpolate(boundary_target, size=logits.shape[2:], mode='nearest')

                    # **Router boundary supervision / KVR router 边界监督**
                    loss_kvr_router += F.binary_cross_entropy_with_logits(
                        logits, curr_target, pos_weight=bce_pos_weight
                    )

                    # **Budget regularization keeps the selected key-voxel ratio near the configured target / 预算约束使关键体素比例接近配置目标**
                    prob = torch.sigmoid(logits)
                    r = base_model.key_voxel_ratio[stage_idx]
                    loss_kvr_budget += (prob.mean() - r) ** 2

            # =========== Total Loss ===========
            # **KVR auxiliary losses are applied throughout training / KVR 辅助损失在整个训练过程中生效**
            base_loss = (fuse_loss * 0.0 + sep_loss + prm_loss) if (epoch < args.region_fusion_start_epoch) else (fuse_loss + sep_loss + prm_loss)
            loss = base_loss + args.lambda_kvr_router * loss_kvr_router + args.lambda_kvr_budget * loss_kvr_budget

            # **Store scalar loss values only to avoid holding autograd graphs / 仅保存标量 loss，避免保留 autograd graph**
            loss_epoch += loss.item()
   
            kvr_router_loss_epoch += loss_kvr_router.item()
            kvr_budget_loss_epoch += loss_kvr_budget.item()

            ### backpropagation
            optimizer.zero_grad()
            loss.backward()

            if not check_grads_and_params(base_model):
                logging.info("[Numerical Debug] Non-finite grad detected: skip optimizer.step() for this iteration.")
                optimizer.zero_grad(set_to_none=True)
                continue


            # **Gradient clipping follows the KVR gamma schedule / 梯度裁剪强度随 KVR gamma 调度变化**
            if epoch < args.kvr_gamma_start_epoch:
                max_norm = 1.0
            elif epoch < (args.kvr_gamma_start_epoch + args.kvr_gamma_warmup_epochs):
                max_norm = 0.3
            else:
                max_norm = 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
 
            optimizer.step()


            current_lr = optimizer.param_groups[0]['lr']


            ### log
            msg = 'Epoch {}/{}, Iter {}/{}, LR {:.6f}, Loss {:.4f}, '.format(
                (epoch+1), args.num_epochs, (i+1), iter_per_epoch, current_lr, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += ' KVRRouter:{:.4f}, KVRBudget:{:.4f}'.format(loss_kvr_router.item(), loss_kvr_budget.item())
            logging.info(msg)

        # **Record throughput-related metrics for monitoring / 记录训练速度相关指标**
        epoch_seconds = time.time() - b
        iter_seconds = epoch_seconds / max(1, iter_per_epoch)
        elapsed_hours = (time.time() - start) / 3600.0
        logging.info('train time per epoch: {}'.format(epoch_seconds))
        lr_schedule.step()

        ########## log current epoch metrics and save current model 
        wandb.log({
            "train/epoch": epoch,
            "train/loss": loss_epoch / iter_per_epoch,
            "train/fusecross": fuse_cross_loss_epoch / iter_per_epoch,
            "train/fusedice": fuse_dice_loss_epoch / iter_per_epoch,
            "train/sepcross": sep_cross_loss_epoch / iter_per_epoch,
            "train/sepdice": sep_dice_loss_epoch / iter_per_epoch,
            "train/prmcross": prm_cross_loss_epoch / iter_per_epoch,
            "train/prmdice": prm_dice_loss_epoch / iter_per_epoch,
            # **KVR auxiliary loss monitoring / KVR 辅助损失监控**
            "train/kvr_router_loss": kvr_router_loss_epoch / iter_per_epoch,
            "train/kvr_budget_loss": kvr_budget_loss_epoch / iter_per_epoch,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            # **Training speed monitoring / 训练速度监控**
            "time/epoch_seconds": epoch_seconds,
            "time/iter_seconds": iter_seconds,
            "time/elapsed_hours": elapsed_hours,
        })

        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'lr_scheduler': lr_schedule.state_dict(),
            'val_Dice_best': val_Dice_best,
            },
            file_name)
        
        ########## validation and test
        if epoch+1 in val_check or args.debug:
            print('validate ...')
            print("[RoPE Check] Expect bottleneck tokens N=512 (8*8*8). If assertion fails later, patch_size may have changed.",
          flush=True)
            sys.stdout.flush()
                
            model.eval()
            
            with torch.no_grad():

                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname,
                    compute_loss = True)
                

                sys.stdout.flush()
        
            val_WT, val_TC, val_ET, val_ETpp = dice_score 
            

            logging.info('Validate epoch = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}, loss = {:.4f}'.format(
                epoch, val_WT, val_TC, val_ET, val_ETpp, seg_loss))
            
            val_dice = (val_ET + val_WT + val_TC)/3
            

            wandb.log({
                "val/epoch": epoch,
                "val/val_ET_Dice": val_ET,
                "val/val_ETpp_Dice": val_ETpp,
                "val/val_WT_Dice": val_WT,
                "val/val_TC_Dice": val_TC,
                "val/val_Dice": val_dice, 
                "val/seg_loss": seg_loss,
            })
            
            if val_dice > val_Dice_best:
                val_Dice_best = val_dice
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'lr_scheduler': lr_schedule.state_dict(),
                    'val_Dice_best': val_Dice_best,
                    },
                    file_name)
                
            print('testing ...')
            sys.stdout.flush() 

            with torch.no_grad():

                dice_score, seg_loss = test_softmax(
                    test_loader,
                    model,
                    dataname = args.dataname,
                    compute_loss = True)
                
                sys.stdout.flush()
                    
            test_WT, test_TC, test_ET, test_ETpp = dice_score   
            
            logging.info('Testing epoch = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ET_postpro = {:.4f}'.format(
                epoch, test_WT, test_TC, test_ET, test_ETpp))
            
            test_dice = (test_ET + test_WT + test_TC)/3
            
            wandb.log({
                "test/epoch": epoch,
                "test/test_WT_Dice": test_WT,
                "test/test_TC_Dice": test_TC,
                "test/test_ET_Dice": test_ET,
                "test/test_ETpp": test_ETpp,
                "test/test_Dice": test_dice,  
                "test/seg_loss": seg_loss,   
            })

            model.train()



    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ########## Evaluate the last epoch model over all modality masks ##########

    print("\n" + "="*30 + " Final Evaluation over 15 Modality Masks " + "="*30)
    test_score = AverageMeter()
    
    with torch.no_grad():
        logging.info('########### test set wi/wo postprocess ###########')
        
        # **Evaluate all predefined missing-modality masks / 评估全部预定义缺失模态 mask**
        for i, mask in enumerate(masks):
            logging.info('Evaluating Mask: {}'.format(mask_name[i]))
            print(f">>> Testing Mask: {mask_name[i]} ...")
            
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask)
            
            # **dice_score stores mean WT, TC, ET, and ETpp scores / dice_score 保存 WT、TC、ET、ETpp 的平均分**
            test_score.update(dice_score)
            
            logging.info('Mask: {} | WT: {:.4f}, TC: {:.4f}, ET: {:.4f}, ETpp: {:.4f}'.format(
                mask_name[i], dice_score[0], dice_score[1], dice_score[2], dice_score[3]))

        # **Report the average over all modality-mask settings / 汇报全部模态 mask 设置下的平均结果**
        avg_scores = test_score.avg
        logging.info('Avg scores over 15 masks: WT: {:.4f}, TC: {:.4f}, ET: {:.4f}, ETpp: {:.4f}'.format(
            avg_scores[0], avg_scores[1], avg_scores[2], avg_scores[3]))
        print("="*80)

    wandb.finish()

if __name__ == '__main__':
    main()
