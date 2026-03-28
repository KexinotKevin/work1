import os
import numpy as np
import nibabel as nib

# 导入第一阶段和第二阶段的自定义函数
from stage1_intra_atlas import (
    load_haufe_weights,
    percentile_rank_transform,
    calculate_stability_score,
    edge_to_node_mapping
)
from stage2_inter_atlas import (
    project_to_voxel_space,
    calculate_gci
)

def main():
    # ---------- 配置参数 ----------
    STATS_DIR = './stats'
    TASK_NAME = 'CogFluidComp_Unadj'
    DATASET = 'S1200'
    ATLAS = 'schaefer200_S1'
    
    # 直接使用您 cfg 中的本地图谱物理路径
    ATLAS_DIR = "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"
    # 拼接出具体的 .nii.gz 文件路径 (假设命名规则为 名字.nii.gz)
    atlas_nifti_path = os.path.join(ATLAS_DIR, f"{ATLAS}.nii.gz")
    
    print(f"=== 运行跨分区稳定性评估流程: {DATASET} | {ATLAS} ===")
    
    # ------------------ 【阶段 1: 特征标准化与秩聚合】 ------------------
    # 1. 加载权重
    weight_matrix, file_names = load_haufe_weights(STATS_DIR, TASK_NAME, DATASET, ATLAS)
    print(f"[阶段1] 加载了 {len(file_names)} 个文件，提取到连接边数: {weight_matrix.shape[1]}")
    
    # 2. 秩转换
    rank_matrix = percentile_rank_transform(weight_matrix)
    
    # 3. 稳定得分
    stability_scores = calculate_stability_score(rank_matrix)
    
    # 4. 节点映射 (选取前 5% 的显著连线)
    nodal_strengths, num_nodes = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)
    print(f"[阶段1] 自适应反推图谱节点数 N = {num_nodes}。")
    
    # ------------------ 【阶段 2: 空间概率映射与 GCI 计算】 ------------------
    if not os.path.exists(atlas_nifti_path):
        raise FileNotFoundError(f"找不到 NIfTI 图谱文件，请检查路径: {atlas_nifti_path}")
        
    print(f"\n[阶段2] 正在加载本地图谱: {atlas_nifti_path}")
    
    # 1. 投射到 3D 体素空间
    volume_data, affine = project_to_voxel_space(nodal_strengths, atlas_nifti_path)
    print(f"[阶段2] 成功投影至体素空间，输出空间矩阵形状: {volume_data.shape}")
    
    # 2. GCI 计算
    # 这里我们模拟跨图谱聚合。即便当前列表里只有一个 volume_data，
    # calculate_gci 也会正确地完成单图谱的 Z-score 脑内归一化。
    gci_volume = calculate_gci([volume_data])
    
    print(f"[阶段2] 全局一致性指数 (GCI) 计算完成。")
    print(f"        大脑内非零体素平均 Z-score (理论上应接近 0): {np.mean(gci_volume[gci_volume != 0]):.4f}")
    print(f"        大脑内最大体素 Z-score (最核心解剖区域): {np.max(gci_volume):.4f}")
    
    # ------------------ 【保存最终结果】 ------------------
    output_dir = os.path.join(STATS_DIR, 'Stage2_Results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带有正确空间坐标系信息的 NIfTI 文件
    out_nifti_path = os.path.join(output_dir, f'GCI_{DATASET}_{ATLAS}.nii.gz')
    gci_img = nib.Nifti1Image(gci_volume, affine)
    nib.save(gci_img, out_nifti_path)
    print(f"\n[完成] 标准化 GCI 脑图已保存至: {out_nifti_path}")

if __name__ == "__main__":
    main()