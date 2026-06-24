import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc
from utils import Parser,criterions2 as criterions

#cudnn.benchmark = True

path = os.path.dirname(__file__)
from utils.generate import generate_snapshot

patch_size = 128

def mask_modal(x, mask):
    # **Apply the modality-availability mask to a stacked modality tensor / 对堆叠模态张量应用可用模态 mask**
    y = torch.zeros_like(x)
    y[mask, ...] = x[mask, ...]
    return y

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    # **Class-wise Dice for BraTS 4-label setting / BraTS 四类别设置下的逐类 Dice**
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1
    
    # **Label 2: edema / 标签 2：水肿区域**
    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2
    
    # **Label 3: enhancing tumor / 标签 3：增强肿瘤区域**
    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    # **ET post-processing metric: suppress tiny enhancing predictions / ET 后处理指标：抑制过小的增强肿瘤预测**
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    # **Class-wise Dice for BraTS 5-label setting / BraTS 五类别设置下的逐类 Dice**
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    # **ET post-processing metric: suppress tiny enhancing predictions / ET 后处理指标：抑制过小的增强肿瘤预测**
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def test_softmax(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        compute_loss=True,
        save_masks=False,
        save_dir=None,
        index=0):

    H, W, T = 240, 240, 155
    loss = 0.0
    model.module.is_training=False
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        
    for i, data in enumerate(test_loader): 
        # **Expected batch fields: image, target, modality mask, one-hot target, case name / batch 字段依次为图像、标签、模态 mask、one-hot 标签、病例名**
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        yo = data[3].cuda()

        # ================== Automatic padding for sliding-window inference / 滑窗推理的自动 padding ==================
        # **Record the original size so padded predictions can be cropped back / 记录原始尺寸，便于最后裁回 padding 前大小**
        _, _, h_orig, w_orig, z_orig = x.size()
        
        # **Pad volumes smaller than patch_size along each spatial axis / 当任一空间轴小于 patch_size 时进行补零**
        pad_h = max(0, patch_size - h_orig)
        pad_w = max(0, patch_size - w_orig)
        pad_z = max(0, patch_size - z_orig)
        
        # **F.pad order is (z_left, z_right, w_left, w_right, h_left, h_right) / F.pad 顺序为 z、w、h 轴的前后补边**
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            x = F.pad(x, (0, pad_z, 0, pad_w, 0, pad_h), mode='constant', value=0)
        # ============================================================

        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()

        # **Use the padded size for sliding-window coordinates / 使用 padding 后的尺寸生成滑窗坐标**
        _, _, H, W, Z = x.size()

        ######### **Generate 50%-overlap sliding-window start indices / 生成 50% 重叠的滑窗起点**
        h_cnt = int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size)

        w_cnt = int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size)

        z_cnt = int(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size)

        ##### **Count how many windows cover each voxel for overlap averaging / 统计每个体素被多少个窗口覆盖，用于重叠区域平均**
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        ##### **Accumulate logits from all sliding-window predictions / 累加所有滑窗预测得到的 logits**
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda() #(B, 4, 133, 176, 135)

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                    # **Enforce the missing-modality mask before model inference / 模型推理前再次强制应用缺失模态 mask**
                    # **Broadcast mask: (B, 4) -> (B, 4, 1, 1, 1) / 广播 mask: (B, 4) -> (B, 4, 1, 1, 1)**
                    mask_broadcast = mask.view(mask.size(0), 4, 1, 1, 1).float()
                    x_input = x_input * mask_broadcast 
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
        pred = pred / weight
        b = time.time()
        # **Crop predictions back to the original image size / 将预测裁回原始图像尺寸**
        pred = pred[:, :, :h_orig, :w_orig, :z_orig]

        
        # **Optional segmentation loss for validation/testing / 验证或测试阶段可选的分割损失**
        if compute_loss:
            # **Cross-entropy consumes logits, matching the training path / 交叉熵直接使用 logits，与训练路径一致**
            seg_cross_loss = criterions.softmax_weighted_loss(pred, yo, num_cls=num_cls)
            
            # **Dice loss consumes probabilities, so logits are converted by softmax / Dice loss 使用概率，因此先对 logits 做 softmax**
            pred_prob = torch.softmax(pred, dim=1)
            seg_dice_loss = criterions.dice_loss(pred_prob, yo, num_cls=num_cls)
            
            seg_loss = seg_cross_loss + seg_dice_loss
            loss += seg_loss

        pred = torch.argmax(pred, dim=1) #(B, 133, 176, 135)

        if dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
            
            # **Use the case name as the default output identifier / 默认使用病例名作为输出标识**
            case_name = name
            out_name = case_name


            # **Optionally save predicted segmentation masks as NIfTI files / 可选将预测分割 mask 保存为 NIfTI 文件**
            if save_masks and save_dir is not None: 
                flags_bool = mask[k].bool().cpu().numpy().tolist()
                flag_str = ''.join(['1' if f else '0' for f in flags_bool])  # -> "0001"

                out_name = f"{case_name}_{flag_str}.nii.gz"
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, out_name)


                # **Use a simple 1.0 mm isotropic affine for saved masks / 保存 mask 时使用 1.0 mm 各向同性 affine**
                affine = np.diag([1.0, 1.0, 1.0, 1.0])

                # **Export one case volume as uint8 segmentation labels / 将单个病例预测导出为 uint8 分割标签**
                pred_np = pred[k].cpu().numpy().astype(np.uint8)  # shape (H, W, T)

                nib.save(nib.Nifti1Image(pred_np, affine), out_path)

            # **Write per-case WT/TC/ET scores and their mean / 写入单病例 WT、TC、ET 分数及其平均值**
            case_scores = scores_evaluation[k][0:3]
            avg_score = float(np.mean(case_scores))

            if save_dir is not None:
                txt_path = os.path.join(save_dir, f"scores_{index}.txt")
                with open(txt_path, "a") as f:
                    f.write(
                        f"{out_name} "
                        + " ".join([f"{s:.4f}" for s in case_scores]) + " "
                        + f"{avg_score:.4f}\n"
                    )
        

    
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print (msg)
    if compute_loss:
        return vals_evaluation.avg, loss/(i+1)
    else:
        return vals_evaluation.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        val_arr = np.asarray(val, dtype=float)
        self.val = val_arr
        if self.count == 0:
            self.sum = val_arr * n
        else:
            self.sum = self.sum + val_arr * n
        self.count += n
        self.avg = self.sum / self.count


