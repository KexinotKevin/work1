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
    calculate_gci
)

def main():
    STATS_DIR = './stats'
    TASK_NAME = 'CogFluidComp_Unadj'
    DATASET = 'S1200'
    
    # 定義需要共同分析的多個圖譜列表
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"
    
    print(f"=== 開始運行多圖譜跨分區穩定性評估流程 (含自動空間對齊) ===")
    
    all_volumes = []
    reference_atlas_path = None # 用於記錄標準空間坐標系
    reference_affine = None     # 用於保存最終的 GCI NIfTI 文件
    
    for atlas_name in ATLASES:
        print(f"\n---> 正在處理圖譜: {atlas_name}")
        atlas_nifti_path = os.path.join(ATLAS_DIR, f"{atlas_name}.nii.gz")
        
        if not os.path.exists(atlas_nifti_path):
            raise FileNotFoundError(f"找不到 NIfTI 圖譜文件: {atlas_nifti_path}")
            
        # 1. 將第一個圖譜設定為基準參考空間
        if reference_atlas_path is None:
            reference_atlas_path = atlas_nifti_path
            print(f"     [空間設置] 已將 {atlas_name} 設定為全局基準坐標系。")
            
        # 2. 加載圖譜（底層會自動檢查並執行 Resampling）
        aligned_atlas_img = load_and_align_atlas(atlas_nifti_path, reference_nifti_path=reference_atlas_path)
        
        if reference_affine is None:
            reference_affine = aligned_atlas_img.affine
            
        # 3. [階段 1]: 第一階段計算
        weight_matrix, file_names = load_haufe_weights(STATS_DIR, TASK_NAME, DATASET, atlas_name)
        rank_matrix = percentile_rank_transform(weight_matrix)
        stability_scores = calculate_stability_score(rank_matrix)
        nodal_strengths, num_nodes = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)
        
        # 4. [階段 2-前期]: 將得分投影到已經對齊的標準體素空間中
        volume_data, affine = project_to_voxel_space(nodal_strengths, aligned_atlas_img)
        print(f"     [階段2] 成功投影至對齊後的體素空間，當前矩陣形狀: {volume_data.shape}")
        
        all_volumes.append(volume_data)
            
    # ---------- 【核心階段 2-後期】: 跨圖譜 GCI 計算 ----------
    print("\n---> 正在執行跨圖譜融合 (Inter-atlas Convergence) ...")
    # 因為前面已經保證了所有 volume_data 形狀絕對一致，這裡的 numpy.mean 不會再拋出維度爆炸的錯誤
    gci_volume = calculate_gci(all_volumes)
    
    print(f"[階段2] 全局一致性指數 (GCI) 融合完成。參與融合圖譜數: {len(all_volumes)}")
    
    # ---------- 保存最終結果 ----------
    output_dir = os.path.join(STATS_DIR, 'Stage2_Results')
    os.makedirs(output_dir, exist_ok=True)
    
    out_nifti_path = os.path.join(output_dir, f'GCI_{DATASET}_Merged_{len(ATLASES)}Atlases.nii.gz')
    gci_img = nib.Nifti1Image(gci_volume, reference_affine)
    nib.save(gci_img, out_nifti_path)
    
    print(f"\n[完成] 跨圖譜融合 GCI 腦圖已保存至: {out_nifti_path}")

if __name__ == "__main__":
    main()