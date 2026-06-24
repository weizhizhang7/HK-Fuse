import torch
import numpy as np
from predict2 import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader
from HKFuse import HKFuse as Model
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--test_file', default='datalist/test15splits.csv', type=str)
parser.add_argument('--datapath', default="/work/grana_neuro/missing_modalities/BRATS2023_Training_npy", type=str)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--kimi_skip', action='store_true', default=False)
parser.add_argument('--kvr_gamma_max', default=0.5, type=float)
parser.add_argument('--disable_kvr', action='store_true', default=False)
parser.add_argument('--kvr_warmup_epochs', default=20, type=int)
parser.add_argument('--kvr_gamma_start_epoch', default=20, type=int)
parser.add_argument('--kvr_gamma_warmup_epochs', default=20, type=int)
parser.add_argument('--force_eval_epoch', default=None, type=int, help='Override epoch used to derive sparse gamma; defaults to checkpoint epoch.')
#parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.savepath:
        os.makedirs(args.savepath, exist_ok=True)
        print(f">>> [Auto-Setup] Created directory: {args.savepath}")
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
    
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = args.datapath
    test_file = args.test_file
    save_path = args.savepath
    num_cls = 4
    dataname = args.dataname
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = Model(
                num_cls=num_cls,
                interleaved_tokenization=args.interleaved_tokenization,
                kimi_skip=args.kimi_skip
            )
    model = torch.nn.DataParallel(model).cuda()

    base_model = model.module if hasattr(model, "module") else model
    if hasattr(base_model, "kvr_gamma_max"):
        base_model.kvr_gamma_max = float(args.kvr_gamma_max)
    if args.disable_kvr and hasattr(base_model, "enable_kvr"):
        base_model.enable_kvr = False
    if hasattr(base_model, "kvr_warmup_epochs"):
        base_model.kvr_warmup_epochs = int(args.kvr_warmup_epochs)
        base_model.kvr_gamma_start_epoch = int(args.kvr_gamma_start_epoch)
        base_model.kvr_gamma_warmup_epochs = int(args.kvr_gamma_warmup_epochs)

    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    skip_keys = []
    cleaned_state_dict = {}
    for key, tensor in state_dict.items():
        if key.endswith("hilbert_idx") or key.endswith("inverse_hilbert_idx"):
            skip_keys.append(key)
            continue
        cleaned_state_dict[key] = tensor
    if skip_keys:
        print(f">>> [Warning] 跳过 Hilbert 缓存参数加载，共 {len(skip_keys)} 项。")
    load_info = model.load_state_dict(cleaned_state_dict, strict=False)
    if load_info.missing_keys:
        print(f">>> [Info] 缺失权重: {load_info.missing_keys}")
    if load_info.unexpected_keys:
        print(f">>> [Info] 多余权重: {load_info.unexpected_keys}")
    best_epoch = checkpoint['epoch'] + 1
    eval_epoch = args.force_eval_epoch if args.force_eval_epoch is not None else best_epoch
    if hasattr(base_model, "set_train_epoch"):
        base_model.set_train_epoch(eval_epoch)

    output_path = None
    if save_path:
        output_path = os.path.join(save_path, f"metrics_K200_epoch{best_epoch}_rank{index}.txt")

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False,
                            save_masks=False,
                            save_dir=save_path,
                            index = index)
            val_WT, val_TC, val_ET, val_ETpp = dice_score

            if output_path:
                with open(output_path, 'a') as file:
                    file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(
                        mask, float(val_WT), float(val_TC), float(val_ET), float(val_ETpp)))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        if output_path:
            with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))
