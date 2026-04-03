import os
import numpy as np
import nibabel as nib
from datasets_cfg import datasets

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

# === run_full_stages.py 修改内容 ===
# 请在文件头部确保导入了 pandas
import pandas as pd

def main(dataset_name=None):
    DATASET = dataset_name if dataset_name is not None else 'S1200'
    tasks = datasets[DATASET]['tgt_label_list'][3:]
    # 【修改1】：对齐新的基准路径
    BASE_DIR = f'./results_CVCR/{DATASET}' 
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "../../datasets/utils/mergedAtlas/Lin6/"
    MODEL_GROUPS = ['linear', 'nonlinear', 'mlp']

    for TASK_NAME in tasks:
        # 【修改4】：跨图谱的公共结果保存在 Label 目录下专门的 Stage_Results 文件夹中
        stage_out_dir = os.path.join(BASE_DIR,'CrossMethodAnal_Results',TASK_NAME)
        os.makedirs(stage_out_dir, exist_ok=True)

        # 【新增】：用于记录所有的 Dice 对齐分数
        dice_records = []

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

                # 获取图谱并提取 Dice 分数
                aligned_atlas_img, dice_score = load_and_align_atlas(atlas_nifti_path, reference_nifti_path=reference_atlas_path)
                if reference_affine is None: reference_affine = aligned_atlas_img.affine

                # 【新增】：记录 Dice 分数
                dice_records.append({
                    "Model_Group": model_group,
                    "Atlas": atlas_name, 
                    "Reference": ATLASES[0], 
                    "Dice_Score": dice_score
                })

                # 【修改2】：遵循 Dataset -> Label -> Atlas -> stats 的新目录层级
                atlas_stats_dir = os.path.join(BASE_DIR, atlas_name, "stats", TASK_NAME)

                weight_matrix, file_names = load_model_weights(
                    atlas_stats_dir, TASK_NAME, DATASET, atlas_name, model_group=model_group
                )

                if len(weight_matrix) == 0:
                    continue

                print(f"     [Stage 1] 圖譜 {atlas_name}: 成功加載 {len(file_names)} 個 {model_group} 模型權重文件。")

                rank_matrix = percentile_rank_transform(weight_matrix)
                stability_scores = calculate_stability_score(rank_matrix)
                nodal_strengths, _ = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)

                # 【修改3】：将 Connectome 级别和 ROI 级别的结果落盘为 .npy，方便后续 draw 画图
                np.save(os.path.join(atlas_stats_dir, f"connectome_stability_{model_group}.npy"), stability_scores)
                np.save(os.path.join(atlas_stats_dir, f"roi_strengths_{model_group}.npy"), nodal_strengths)
                print(f"     [Save] 已保存 {atlas_name} 的 Connectome 和 ROI 矩阵。")

                volume_data, _ = project_to_voxel_space(nodal_strengths, aligned_atlas_img)
                all_volumes.append(volume_data)

            if not all_volumes:
                print(f"⚠️ 跳過 [{model_group.upper()}] 分析，因為未提取到任何有效數據。")
                continue

            # ---------------- Stage 2 後期：GCI 計算 ----------------
            gci_volume, confidence_volume = calculate_gci_and_confidence(all_volumes)


            
            gci_nii_path = os.path.join(stage_out_dir, f'GCI_{DATASET}_{model_group}_Merged_{len(ATLASES)}Atlases.nii.gz')
            nib.save(nib.Nifti1Image(gci_volume, reference_affine), gci_nii_path)
            
            # 【新增】：额外保存一份 Numpy 矩阵版本的 GCI 和 Confidence，方便纯 Python 画 Heatmap
            np.save(os.path.join(stage_out_dir, f'GCI_matrix_{model_group}.npy'), gci_volume)
            np.save(os.path.join(stage_out_dir, f'Confidence_matrix_{model_group}.npy'), confidence_volume)
            print(f"     [Stage 2] 跨圖譜融合完成，GCI/Confidence NIfTI與Numpy矩陣已保存。")

            # ---------------- Stage 3：統計推斷 ----------------
            null_distribution = mock_null_distribution()
            fwe_corrected_volume, report = cluster_level_fwe_correction(
                real_gci_volume=gci_volume,
                null_max_sizes=null_distribution,
                primary_threshold=1.0,
                p_val_thresh=0.05
            )

            fwe_path = os.path.join(stage_out_dir, f'GCI_{DATASET}_{model_group}_FWECorrected.nii.gz')
            nib.save(nib.Nifti1Image(fwe_corrected_volume, reference_affine), fwe_path)
            # 【新增】：保存 FWE 统计校正报告
            pd.DataFrame(report).to_csv(os.path.join(stage_out_dir, f'FWE_Report_{model_group}.csv'), index=False)
            print(f"     [Stage 3] 統計校正完成，已保存 NIfTI 與 Report CSV。")

        # 【修改5】：在最外层循环结束后，统一保存 Dice 分数文件
        if dice_records:
            dice_df = pd.DataFrame(dice_records).drop_duplicates()
            dice_csv_path = os.path.join(stage_out_dir, 'alignment_dice_scores.csv')
            dice_df.to_csv(dice_csv_path, index=False)
            print(f"✅ 全部分支執行完畢，圖譜對齊 Dice 分數已保存至: {dice_csv_path}")

if __name__ == "__main__":
    main(dataset_name='ABCD')