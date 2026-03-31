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
    BASE_DIR = f'./res_{DATASET}'
    TASK_NAME = 'CogFluidComp_Unadj' if task_name is None else task_name
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"

    # 【核心修改 1】：引入全新的三個演算法分組
    MODEL_GROUPS = ['linear', 'nonlinear', 'mlp']

    # 外層循環，將三種不同解釋性框架的結果完全隔離開
    for model_group in MODEL_GROUPS:
        print(f"\n{'='*60}")
        print(f"=== 開始執行全流程計算分支: [{model_group.upper()}] ===")
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

            # 新目录结构: res_{dataset}/{atlas}/stats/
            atlas_stats_dir = os.path.join(BASE_DIR, atlas_name, "stats")

            # 【核心修改 2】：參數名改為 model_group=model_group
            weight_matrix, file_names = load_model_weights(
                atlas_stats_dir, TASK_NAME, DATASET, atlas_name, model_group=model_group
            )

            if len(weight_matrix) == 0:
                continue

            print(f"     [Stage 1] 圖譜 {atlas_name}: 成功加載 {len(file_names)} 個 {model_group} 模型權重文件。")

            rank_matrix = percentile_rank_transform(weight_matrix)
            stability_scores = calculate_stability_score(rank_matrix)
            nodal_strengths, _ = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)

            volume_data, _ = project_to_voxel_space(nodal_strengths, aligned_atlas_img)
            all_volumes.append(volume_data)

        if not all_volumes:
            print(f"⚠️ 跳過 [{model_group.upper()}] 分析，因為未提取到任何有效數據。")
            continue

        # ---------------- Stage 2 後期：GCI 計算 ----------------
        gci_volume, confidence_volume = calculate_gci_and_confidence(all_volumes)

        # 新目录结构: res_{dataset}/{atlas}/stats/{task}/Stage2_Results
        stage2_out_dir = os.path.join(BASE_DIR, ATLASES[0], "stats", TASK_NAME, 'Stage2_Results')
        os.makedirs(stage2_out_dir, exist_ok=True)
        gci_path = os.path.join(stage2_out_dir, f'GCI_{DATASET}_{model_group}_Merged_{len(ATLASES)}Atlases.nii.gz')
        nib.save(nib.Nifti1Image(gci_volume, reference_affine), gci_path)
        print(f"     [Stage 2] 跨圖譜融合完成，已保存至: {gci_path}")

        # ---------------- Stage 3：統計推斷 ----------------
        null_distribution = mock_null_distribution()
        fwe_corrected_volume, report = cluster_level_fwe_correction(
            real_gci_volume=gci_volume,
            null_max_sizes=null_distribution,
            primary_threshold=1.0,
            p_val_thresh=0.05
        )

        # 新目录结构: res_{dataset}/{atlas}/stats/{task}/Stage3_Results
        stage3_out_dir = os.path.join(BASE_DIR, ATLASES[0], "stats", TASK_NAME, 'Stage3_Results')
        os.makedirs(stage3_out_dir, exist_ok=True)
        fwe_path = os.path.join(stage3_out_dir, f'GCI_{DATASET}_{model_group}_FWECorrected.nii.gz')
        nib.save(nib.Nifti1Image(fwe_corrected_volume, reference_affine), fwe_path)
        print(f"     [Stage 3] 統計校正完成，已保存至: {fwe_path}")

        print(f"✅ [{model_group.upper()}] 分支執行完畢，計算結果已完全隔離並落盤。")

if __name__ == "__main__":
    main()