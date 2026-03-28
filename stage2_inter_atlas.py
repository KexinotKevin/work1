import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

# 更新：增加了 Dice 分数计算以衡量空间对齐的质量，并在加载图谱时自动执行对齐和质量评估。

def calculate_dice(mask1, mask2):
    """辅助函数：计算两个布尔掩码的 Dice 相似系数"""
    intersection = np.logical_and(mask1, mask2).sum()
    total_elements = mask1.sum() + mask2.sum()
    if total_elements == 0:
        return 0.0
    return 2. * intersection / total_elements

def load_and_align_atlas(atlas_nifti_path, reference_nifti_path=None):
    """
    加载并对齐图谱，同时计算对齐后的 Dice 分数以衡量空间不确定性。
    """
    atlas_img = nib.load(atlas_nifti_path)
    dice_score = 1.0  # 默认为 1.0 (如果是自身或无需对齐)
    
    if reference_nifti_path is not None:
        ref_img = nib.load(reference_nifti_path)
        
        shape_mismatch = atlas_img.shape != ref_img.shape
        affine_mismatch = not np.allclose(atlas_img.affine, ref_img.affine)
        
        if shape_mismatch or affine_mismatch:
            # 1. 强行重采样到参考空间 (使用 nearest 保证标签整数不被破坏)
            atlas_img = resample_to_img(
                source_img=atlas_img, 
                target_img=ref_img, 
                interpolation="nearest"
            )
            
            # 2. 计算宏观不确定性衡量值：Dice 分数
            # 提取大脑有效区域掩码 (非零区域)
            ref_mask = ref_img.get_fdata() != 0
            aligned_mask = atlas_img.get_fdata() != 0
            dice_score = calculate_dice(ref_mask, aligned_mask)
            
    return atlas_img, dice_score

def project_to_voxel_space(nodal_strengths, atlas_img):
    """空间概率贡献地图投影 (逻辑不变)"""
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    labels = np.unique(atlas_data)
    labels = labels[labels != 0]
    
    if len(nodal_strengths) != len(labels):
        raise ValueError(f"维度不匹配: 得分数({len(nodal_strengths)}) 与 标签数({len(labels)}) 不一致！")
    
    volume_data = np.zeros_like(atlas_data, dtype=np.float32)
    for idx, label in enumerate(labels):
        volume_data[atlas_data == label] = nodal_strengths[idx]
        
    return volume_data, affine

def calculate_gci_and_confidence(volume_list):
    """
    计算全局一致性指数 (GCI) 以及体素级置信度地图 (Confidence Map)。
    """
    z_scored_volumes = []
    valid_masks = []
    
    for volume in volume_list:
        brain_mask = volume != 0
        valid_masks.append(brain_mask)
        
        if not np.any(brain_mask):
            z_volume = np.zeros_like(volume)
        else:
            mean_val = np.mean(volume[brain_mask])
            std_val = np.std(volume[brain_mask])
            z_volume = np.zeros_like(volume, dtype=np.float32)
            z_volume[brain_mask] = (volume[brain_mask] - mean_val) / (std_val + 1e-8)
            
        z_scored_volumes.append(z_volume)
    
    z_scored_volumes = np.array(z_scored_volumes) # 形状: (T, X, Y, Z)
    valid_masks = np.array(valid_masks)           # 形状: (T, X, Y, Z)
    
    # 1. 计算置信度地图 (Confidence Map)
    # 统计每个体素处，有多少个图谱提供了有效数据 (非 0)
    valid_counts = np.sum(valid_masks, axis=0)
    T_total = len(volume_list)
    confidence_volume = valid_counts.astype(np.float32) / T_total
    
    # 2. 修正后的 GCI 计算 (排除缺失覆盖的图谱)
    # 以前是无脑除以 T_total，现在我们只除以真正覆盖了该体素的图谱数量，避免 GCI 被背景 0 强行拉低
    with np.errstate(divide='ignore', invalid='ignore'):
        gci_volume = np.sum(z_scored_volumes, axis=0) / valid_counts
        # 对于所有图谱都没有覆盖的背景区域，设为 0
        gci_volume[np.isnan(gci_volume)] = 0.0
        
    return gci_volume, confidence_volume