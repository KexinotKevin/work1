import os
import numpy as np
import nibabel as nib

from stage1_intra_atlas import (
    load_model_weights, # 注意这里导入了修改后的新函数名
    percentile_rank_transform, 
    calculate_stability_score, 
    edge_to_node_mapping
)
from stage2_inter_atlas import load_and_align_atlas, project_to_voxel_space, calculate_gci_and_confidence
from stage3_statistical_inference import cluster_level_fwe_correction

# 模拟零分布（实际应用中需替换为真实置换检验结果）
def mock_null_distribution():
    return np.sort(np.random.exponential(scale=50, size=10000).astype(int))

def main(task_name=None):
    DATASET = 'S1200'
    STATS_DIR = f'./res_{DATASET}/stats'
    TASK_NAME = 'CogFluidComp_Unadj' if task_name is None else task_name
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"
    
    # 【核心修正】：增加外层循环，将线性和非线性完全隔离开
    MODEL_TYPES = ['linear', 'nonlinear']
    
    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"=== 开始执行分支: [{model_type.upper()}] (特征溯源: {'Haufe' if model_type=='linear' else 'SHAP'}) ===")
        print(f"{'='*60}")
        
        all_volumes = []
        reference_atlas_path = None 
        reference_affine = None     
        
        # ---------------- Stage 1 & 2 前期 ----------------
        for atlas_name in ATLASES:
            atlas_nifti_path = os.path.join(ATLAS_DIR, f"{atlas_name}.nii.gz")
            if reference_atlas_path is None: reference_atlas_path = atlas_nifti_path
                
            aligned_atlas_img, _ = load_and_align_atlas(atlas_nifti_path, reference_nifti_path=reference_atlas_path)
            if reference_affine is None: reference_affine = aligned_atlas_img.affine
                
            # 调用修正后的函数并传入 model_type
            weight_matrix, _ = load_model_weights(STATS_DIR, TASK_NAME, DATASET, atlas_name, model_type=model_type)
            
            if len(weight_matrix) == 0:
                continue
                
            rank_matrix = percentile_rank_transform(weight_matrix)
            stability_scores = calculate_stability_score(rank_matrix)
            nodal_strengths, _ = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)
            
            volume_data, _ = project_to_voxel_space(nodal_strengths, aligned_atlas_img)
            all_volumes.append(volume_data)
            
        if not all_volumes:
            print(f"跳过 {model_type} 分析，因为未提取到有效数据。")
            continue

        # ---------------- Stage 2 后期：GCI 计算 ----------------
        gci_volume, confidence_volume = calculate_gci_and_confidence(all_volumes)
        
        # 【保存修正】：在文件名中显式写入 model_type
        stage2_out_dir = os.path.join(STATS_DIR, 'Stage2_Results')
        os.makedirs(stage2_out_dir, exist_ok=True)
        gci_path = os.path.join(stage2_out_dir, f'GCI_{DATASET}_{model_type}_Merged_{len(ATLASES)}Atlases.nii.gz')
        nib.save(nib.Nifti1Image(gci_volume, reference_affine), gci_path)
        
        # ---------------- Stage 3：统计推断 ----------------
        null_distribution = mock_null_distribution()
        fwe_corrected_volume, report = cluster_level_fwe_correction(
            real_gci_volume=gci_volume,
            null_max_sizes=null_distribution,
            primary_threshold=1.0,
            p_val_thresh=0.05
        )
        
        # 【保存修正】：在文件名中显式写入 model_type
        stage3_out_dir = os.path.join(STATS_DIR, 'Stage3_Results')
        os.makedirs(stage3_out_dir, exist_ok=True)
        fwe_path = os.path.join(stage3_out_dir, f'GCI_{DATASET}_{model_type}_FWECorrected.nii.gz')
        nib.save(nib.Nifti1Image(fwe_corrected_volume, reference_affine), fwe_path)
        
        print(f"[{model_type.upper()}] 分支执行完毕，结果已隔离保存。")

if __name__ == "__main__":
    main()