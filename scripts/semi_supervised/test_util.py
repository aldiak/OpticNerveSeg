import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import os
import logging
from networks.my_net import LeFeD_Net


def test_all_case(
    model,
    image_list,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    save_result=True,
    test_save_path=None,
    metric_detail=0,
):
    """
    Test model on all cases in image_list
    """
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    test_set_metrics = []
    
    for image_path in loader:
        h5f = h5py.File(image_path, "r")
        image = h5f["image_t1"][:]
        image1 = h5f["image_fa"][:]
        label = h5f["label"][:]
        
        # Get voxel spacing if available, otherwise use default (1.0, 1.0, 1.0)
        if "voxel_spacing" in h5f.attrs:
            voxel_spacing = h5f.attrs["voxel_spacing"]
        else:
            voxel_spacing = (1.0, 1.0, 1.0)
            logging.warning(f"Voxel spacing not found in {image_path}, using default (1.0, 1.0, 1.0)")
        
        prediction, score_map = test_single_case(
            model,
            image,
            image1,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes,
        )

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            # Pass voxel spacing to metric calculation
            single_metric = calculate_metric_percase(prediction, label[:], voxel_spacing)

        test_set_metrics.append(
            [
                single_metric[0] * 100,
                single_metric[1] * 100,
                single_metric[2],
                single_metric[3],
            ]
        )
        
        logging.info(
            "%02d,\t%.5f, %.5f, %.5f, %.5f"
            % (
                ith,
                single_metric[0],
                single_metric[1],
                single_metric[2],
                single_metric[3],
            )
        )

        total_metric += np.asarray(single_metric)

        if save_result:
            # Create affine matrix with correct voxel spacing
            affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
            
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32), affine),
                test_save_path + "%02d_pred_%.2f.nii.gz" % (ith, single_metric[0]),
            )
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32), affine),
                test_save_path + "%02d_img.nii.gz" % ith,
            )
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32), affine),
                test_save_path + "%02d_gt.nii.gz" % ith,
            )
        ith += 1

    avg_metric = total_metric / len(image_list)
    print("average metric is {}".format(avg_metric))

    mean = np.mean(test_set_metrics, axis=0)
    std_var = np.std(test_set_metrics, axis=0)
    return mean, std_var


def test_single_case(
    net,
    image,
    image1,
    stride_xy,
    stride_z,
    patch_size,
    num_classes=2,
    is_dt=False,
    is_urpc=False,
    is_lfd=False,
):
    """
    Test a single case with sliding window
    """
    w, h, d = image.shape
    
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
        
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
        image1 = np.pad(
            image1,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
    
    ww, hh, dd = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                
                test_patch = image[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0), axis=0
                ).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                test_patch1 = image1[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch1 = np.expand_dims(
                    np.expand_dims(test_patch1, axis=0), axis=0
                ).astype(np.float32)
                test_patch1 = torch.from_numpy(test_patch1).cuda()
                
                input_patch = torch.cat((test_patch, test_patch1), dim=1)

                with torch.no_grad():
                    if is_urpc:
                        # Deep supervision: take only the full-resolution head (dsv1)
                        out, _, _, _ = net(input_patch)
                    elif is_dt:
                        # UGMCL: seg_logits is the second output
                        _, out = net(input_patch)
                    elif is_lfd or isinstance(net, LeFeD_Net):
                        out, y1, _, _, _ = net(input_patch, [])
                    else:
                        # Standard U-Net
                        out = net(input_patch)

                    y = F.softmax(out, dim=1).cpu().numpy()[0]  # [C, H, W, D]

                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    score_map[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + y
                )
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    cnt[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + 1
                )
                
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    
    if add_pad:
        label_map = label_map[
            wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        score_map = score_map[
            :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        
    return label_map, score_map


def test_all_case_all(
    model,
    image_list,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    is_dt=False,
    is_urpc=False,
    is_lfd=False,
    save_result=True,
    test_save_path=None,
    metric_detail=0,
    voxel_spacing=(1.0, 1.0, 1.0)
):
    """
    Test all cases with model type flags
    """
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    test_set_metrics = []
    
    for image_path in loader:
        h5f = h5py.File(image_path, "r")
        image = h5f["image_t1"][:]
        image1 = h5f["image_fa"][:]
        label = h5f["label"][:]
        
        # Get voxel spacing if available, otherwise use default
        if "voxel_spacing" in h5f.attrs:
            voxel_spacing = h5f.attrs["voxel_spacing"]
        else:
            voxel_spacing = voxel_spacing  # Use the provided default
            logging.warning(f"Voxel spacing not found in {image_path}, using default {voxel_spacing}")
        
        prediction, score_map = test_single_case(
            model,
            image,
            image1,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes,
            is_dt=is_dt,
            is_urpc=is_urpc,
            is_lfd=is_lfd,
        )

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            # Pass voxel spacing to metric calculation
            single_metric = calculate_metric_percase(prediction, label[:], voxel_spacing)

        test_set_metrics.append(
            [
                single_metric[0] * 100,
                single_metric[1] * 100,
                single_metric[2],
                single_metric[3],
            ]
        )
        
        logging.info(
            "%02d,\t%.5f, %.5f, %.5f, %.5f"
            % (
                ith,
                single_metric[0],
                single_metric[1],
                single_metric[2],
                single_metric[3],
            )
        )

        total_metric += np.asarray(single_metric)

        if save_result:
            # Create affine matrix with correct voxel spacing
            affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
            
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32), affine),
                test_save_path + "%02d_pred_%.2f.nii.gz" % (ith, single_metric[0]),
            )
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32), affine),
                test_save_path + "%02d_img.nii.gz" % ith,
            )
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32), affine),
                test_save_path + "%02d_gt.nii.gz" % ith,
            )
        ith += 1

    avg_metric = total_metric / len(image_list)
    print("average metric is {}".format(avg_metric))

    mean = np.mean(test_set_metrics, axis=0)
    std_var = np.std(test_set_metrics, axis=0)
    return mean, std_var


def test_single_case_all(
    net,
    image,
    image1,
    stride_xy,
    stride_z,
    patch_size,
    num_classes=2,
    is_dt=False,
    is_urpc=False,
):
    """
    Alternative test single case function with simplified padding
    """
    w, h, d = image.shape

    # Padding
    pad = [(0, max(0, s - x)) for x, s in zip(image.shape, patch_size)]
    pad = [(p // 2, p - p // 2) for p in [sum(p) for p in zip(*pad)]]
    if any(sum(p) > 0 for p in pad):
        image = np.pad(image, pad, mode="constant")
        image1 = np.pad(image1, pad, mode="constant")

    ww, hh, dd = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])

                test_patch = image[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch1 = image1[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]

                test_patch = (
                    torch.from_numpy(test_patch[None, None, ...]).float().cuda()
                )
                test_patch1 = (
                    torch.from_numpy(test_patch1[None, None, ...]).float().cuda()
                )
                input_patch = torch.cat((test_patch, test_patch1), dim=1)

                with torch.no_grad():
                    if is_urpc:
                        out, _, _, _ = net(input_patch)
                    elif is_dt:
                        _, out = net(input_patch)
                    else:
                        out = net(input_patch)

                    out = F.softmax(out, dim=1).cpu().numpy()[0]  # [C, H, W, D]

                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] += out
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] += 1

    score_map /= cnt + 1e-8
    label_map = np.argmax(score_map, axis=0)

    # Remove padding
    if any(sum(p) > 0 for p in pad):
        label_map = label_map[[slice(p[0], -p[1] or None) for p in pad]]

    return label_map, score_map


def cal_dice(prediction, label, num=2):
    """
    Calculate Dice score for each class
    """
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = prediction == i
        label_tmp = label == i
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = (
            2
            * np.sum(prediction_tmp * label_tmp)
            / (np.sum(prediction_tmp) + np.sum(label_tmp))
        )
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculate metrics for a single case with proper voxel spacing
    """
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    
    # Pass voxel spacing to HD95 and ASD for accurate physical distance measurements
    hd = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
    asd = metric.binary.asd(pred, gt, voxelspacing=voxel_spacing)

    return dice, jc, hd, asd


def test_calculate_metric(
    net,
    test_dataset,
    num_classes=2,
    save_result=False,
    test_save_path="./save",
    stride_xy=16,
    stride_z=4,
):
    """
    Test and calculate metrics using test_dataset
    """
    net.eval()
    image_list = test_dataset.test_list_path

    if save_result:
        test_save_path = Path(test_save_path)
        test_save_path.mkdir(exist_ok=True)

    avg_metric = test_all_case(
        net,
        image_list,
        num_classes=num_classes,
        patch_size=(96, 96, 96),
        stride_xy=stride_xy,
        stride_z=stride_z,
        save_result=save_result,
        test_save_path=str(test_save_path) + "/",
    )
    return avg_metric


def test_calculate_metric_LA(
    net,
    ema_net,
    test_dataset,
    num_classes=2,
    save_result=False,
    test_save_path="./save",
):
    """
    Test and calculate metrics for LA dataset
    """
    net.eval()

    image_list = test_dataset.image_list
    avg_metric = test_all_case(
        net,
        ema_net,
        image_list,
        num_classes=num_classes,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        save_result=save_result,
        test_save_path=test_save_path,
    )
    return avg_metric


def create_h5_from_nifti(nifti_path, output_h5_path, modality="t1"):
    """
    Utility function to create HDF5 file from NIfTI with voxel spacing
    Example function to show how to store voxel spacing during dataset creation
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Get voxel spacing from NIfTI header
    voxel_spacing = img.header.get_zooms()[:3]  # (x, y, z) spacing
    
    with h5py.File(output_h5_path, 'w') as h5f:
        if modality == "t1":
            h5f.create_dataset('image_t1', data=data, compression='gzip')
        elif modality == "fa":
            h5f.create_dataset('image_fa', data=data, compression='gzip')
        elif modality == "label":
            h5f.create_dataset('label', data=data, compression='gzip')
        
        # Store voxel spacing as attribute
        h5f.attrs['voxel_spacing'] = voxel_spacing
    
    print(f"Saved {output_h5_path} with voxel spacing: {voxel_spacing}")
    return voxel_spacing


def create_multi_modality_h5(t1_path, fa_path, label_path, output_h5_path):
    """
    Utility function to create multi-modality HDF5 file with consistent voxel spacing
    """
    # Load all modalities
    t1_img = nib.load(t1_path)
    fa_img = nib.load(fa_path)
    label_img = nib.load(label_path)
    
    # Get voxel spacing (assuming all have same spacing)
    voxel_spacing = t1_img.header.get_zooms()[:3]
    
    # Get data
    t1_data = t1_img.get_fdata()
    fa_data = fa_img.get_fdata()
    label_data = label_img.get_fdata()
    
    # Verify shapes match
    assert t1_data.shape == fa_data.shape == label_data.shape, "All modalities must have same shape"
    
    with h5py.File(output_h5_path, 'w') as h5f:
        h5f.create_dataset('image_t1', data=t1_data, compression='gzip')
        h5f.create_dataset('image_fa', data=fa_data, compression='gzip')
        h5f.create_dataset('label', data=label_data, compression='gzip')
        
        # Store voxel spacing as attribute
        h5f.attrs['voxel_spacing'] = voxel_spacing
    
    print(f"Saved {output_h5_path} with voxel spacing: {voxel_spacing}")
    return voxel_spacing