import numpy as np
import nibabel as nib

def project_to_voxel_space(nodal_strengths, atlas_nifti_path):
    """
    第一步：空间概率贡献地图投影 (Voxel-wise Projection)
    将一维的脑区节点得分，映射回三维标准脑空间中对应的体素上。
    
    参数:
        nodal_strengths (np.ndarray): 形状为 (N,) 的一维数组，对应 N 个脑区的得分。
        atlas_nifti_path (str): 脑分区 NIfTI 文件 (.nii 或 .nii.gz) 的绝对路径。
        
    返回:
        volume_data (np.ndarray): 映射后的三维体素矩阵 (3D NumPy Array)。
        affine (np.ndarray): NIfTI 仿射矩阵，用于后续保存为标准脑影像文件。
    """
    # 1. 加载脑分区图像
    atlas_img = nib.load(atlas_nifti_path)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    
    # 2. 获取所有存在的脑区标签 (剔除代表背景的 0)
    # np.unique 会自动从小到大排序，这与脑区 ID 通常的递增顺序一致
    labels = np.unique(atlas_data)
    labels = labels[labels != 0]
    
    # 维度安全校验：确保提取出的连线节点数与图谱实际包含的脑区数一致
    if len(nodal_strengths) != len(labels):
        raise ValueError(f"维度不匹配: 节点得分数量({len(nodal_strengths)}) 与 图谱实际ROI标签数量({len(labels)}) 不一致！")
    
    # 3. 创建空白的全脑 3D 画布 (浮点型，用于保存连续的得分)
    volume_data = np.zeros_like(atlas_data, dtype=np.float32)
    
    # 4. 执行空间投影
    # 遍历每个脑区标签，把对应的得分填入该标签所在的体素区域
    for idx, label in enumerate(labels):
        score = nodal_strengths[idx]
        # 布尔索引：将大脑空间中等于该 label 的体素值全部替换为对应的节点得分
        volume_data[atlas_data == label] = score
        
    return volume_data, affine

def calculate_gci(volume_list):
    """
    第二步：全局一致性指数计算 (Global Consistency Index, GCI)
    对多个脑分区投影产生的三维地图分别进行 Z-score 标准化，并求均值。
    
    参数:
        volume_list (list of np.ndarray): 包含多个 3D 体素矩阵的列表。
        
    返回:
        gci_volume (np.ndarray): 计算完 GCI 后的三维体素矩阵。
    """
    z_scored_volumes = []
    
    for volume in volume_list:
        # 【关键统计学修正】：必须仅对有信号的脑区（非零体素）计算 mean 和 std。
        # 如果包含背景（0），海量的背景像素会严重拉低均值，导致 Z-score 完全失效。
        brain_mask = volume != 0
        
        if not np.any(brain_mask):
            z_volume = np.zeros_like(volume)
        else:
            mean_val = np.mean(volume[brain_mask])
            std_val = np.std(volume[brain_mask])
            
            z_volume = np.zeros_like(volume, dtype=np.float32)
            # 仅对脑区体素进行 Z-score 转换：(x - μ) / σ
            z_volume[brain_mask] = (volume[brain_mask] - mean_val) / (std_val + 1e-8)
            
        z_scored_volumes.append(z_volume)
    
    # 将所有的 Z-score 地图沿新维度堆叠，并计算平均值 (对应公式中的 1/T * Σ)
    z_scored_volumes = np.array(z_scored_volumes)
    gci_volume = np.mean(z_scored_volumes, axis=0)
    
    return gci_volume