import os
import numpy as np
import nibabel as nib

from stage1_intra_atlas import (
    load_haufe_weights,
    percentile_rank_transform,
    calculate_stability_score,
    edge_to_node_mapping
)
from stage2_inter_atlas import (
    load_and_align_atlas,
    project_to_voxel_space,
    calculate_gci_and_confidence
)

def main():
    STATS_DIR = './stats'
    TASK_NAME = 'CogFluidComp_Unadj'
    DATASET = 'S1200'
    
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"
    
    print(f"=== 开始运行多图谱跨分区稳定性评估 (包含不确定性衡量) ===")
    
    all_volumes = []
    reference_atlas_path = None 
    reference_affine = None     
    
    for atlas_name in ATLASES:
        print(f"\n---> 正在处理图谱: {atlas_name}")
        atlas_nifti_path = os.path.join(ATLAS_DIR, f"{atlas_name}.nii.gz")
        
        if reference_atlas_path is None:
            reference_atlas_path = atlas_nifti_path
            print(f"     [空间设置] 将 {atlas_name} 设为全局基准坐标系。")
            
        # 1. 加载并对齐，同时获取 Dice 分数
        aligned_atlas_img, dice_score = load_and_align_atlas(
            atlas_nifti_path, 
            reference_nifti_path=reference_atlas_path
        )
        
        if dice_score == 1.0:
            print(f"     [空间质量] 原生空间或同图谱，无需重采样 (Dice = 1.0)")
        else:
            # 报告宏观对齐质量
            print(f"     [空间质量] 与基准空间的脑区重合度 Dice Score = {dice_score:.4f}")
            if dice_score < 0.8:
                print("     ⚠️ 警告: Dice 分数较低，可能存在较严重的解剖边界不匹配！")
        
        if reference_affine is None:
            reference_affine = aligned_atlas_img.affine
            
        # 2. 走完第一阶段流程
        weight_matrix, _ = load_haufe_weights(STATS_DIR, TASK_NAME, DATASET, atlas_name)
        rank_matrix = percentile_rank_transform(weight_matrix)
        stability_scores = calculate_stability_score(rank_matrix)
        nodal_strengths, _ = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)
        
        # 3. 投影到对齐后的体素空间
        volume_data, affine = project_to_voxel_space(nodal_strengths, aligned_atlas_img)
        all_volumes.append(volume_data)
            
    # ---------- 【计算 GCI 与 置信度】 ----------
    print("\n---> 正在执行跨图谱融合与体素级置信度评估 ...")
    gci_volume, confidence_volume = calculate_gci_and_confidence(all_volumes)
    
    print(f"[完成] 全局一致性指数 (GCI) 融合完成。")
    print(f"       全脑最高 GCI 极值: {np.max(gci_volume):.4f}")
    print(f"       完美覆盖体素 (Confidence=1.0) 占比: {(np.sum(confidence_volume == 1.0) / np.sum(confidence_volume > 0) * 100):.1f}%")
    
    # ---------- 保存最终结果 ----------
    output_dir = os.path.join(STATS_DIR, 'Stage2_Results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 GCI 脑图
    out_gci_path = os.path.join(output_dir, f'GCI_{DATASET}_Merged_{len(ATLASES)}Atlases.nii.gz')
    nib.save(nib.Nifti1Image(gci_volume, reference_affine), out_gci_path)
    
    # 保存 置信度 脑图
    out_conf_path = os.path.join(output_dir, f'ConfidenceMap_{DATASET}_Merged.nii.gz')
    nib.save(nib.Nifti1Image(confidence_volume, reference_affine), out_conf_path)
    
    print(f"\n已保存结果文件:")
    print(f"1. 预测贡献核心热图: {out_gci_path}")
    print(f"2. 空间可信度验证图: {out_conf_path} (值范围 0~1)")

if __name__ == "__main__":
    main()