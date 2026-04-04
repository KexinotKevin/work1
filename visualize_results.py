import os
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_stat_map
from datasets_cfg import datasets

from draw_brain import draw_connectome, draw_atlas_roi, _save_display

def build_adjacency_matrix(stability_scores, threshold_percentile):
    """
    根据保留的稳定分数和阈值百分位，重建 N x N 的邻接矩阵。
    
    支持有符号的 stability_scores：
    - 阈值筛选基于绝对值，确保正负显著边都被考虑
    - 保留原始符号
    
    参数:
        stability_scores: 有符号的边得分数组
        threshold_percentile: 阈值百分位（基于绝对值）
    """
    D = len(stability_scores)
    N = int(np.round((1 + np.sqrt(1 + 8 * D)) / 2))
    
    # 基于绝对值进行阈值筛选
    abs_scores = np.abs(stability_scores)
    threshold_val = np.quantile(abs_scores, threshold_percentile)
    filtered_scores = np.where(abs_scores >= threshold_val, stability_scores, 0.0)
    
    adj_matrix = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    adj_matrix[iu] = filtered_scores
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def main(dataset_name=None):
    DATASET = dataset_name if dataset_name is not None else 'S1200'
    tasks = datasets[DATASET]['tgt_label_list'][3:]
    
    # 【适配修改 1】 指向最新的结果目录
    BASE_DIR = f'./results_CVCR/{DATASET}'
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "../../datasets/utils/mergedAtlas/Lin6/"
    # 模型分组
    MODEL_GROUPS = ['linear', 'nonlinear', 'mlp']
    # 与 run_full_stages 保持对齐的模态列表
    MODALITIES = ['SC', 'FC', 'ALL']

    for TASK_NAME in tasks:
        # 外层模态循环
        for modality in MODALITIES:
            for model_group in MODEL_GROUPS:
                print(f"\n{'='*50}")
                print(f"🚀 开始可视化: [Modality: {modality}] [{model_group.upper()}]")
                print(f"{'='*50}")

            # -----------------------------------------------------------------
            # 需求 1 & 2: 提取同种分区、跨方法的关键连边与关键脑区 Numpy 矩阵
            # -----------------------------------------------------------------
            for atlas_name in ATLASES:
                print(f"\n>>> 正在生成分区绘图: {atlas_name} [Modality: {modality}]")
                atlas_nifti_path = os.path.join(ATLAS_DIR, f"{atlas_name}.nii.gz")

                # 【适配修改 2】 定位由 run_full_stages 落盘的 .npy 路径
                atlas_stats_dir = os.path.join(BASE_DIR, atlas_name, "stats", TASK_NAME)
                # 生成对应图片的输出目录: results_CVCR/{DATASET}/{ATLAS}/pics/{TASK_NAME}/
                atlas_pics_dir = os.path.join(BASE_DIR, atlas_name, "pics", TASK_NAME)
                os.makedirs(atlas_pics_dir, exist_ok=True)

                # 【修改】：读取带 modality 后缀的文件
                stability_npy_path = os.path.join(atlas_stats_dir, f"connectome_stability_{model_group}_{modality}.npy")
                roi_npy_path = os.path.join(atlas_stats_dir, f"roi_strengths_{model_group}_{modality}.npy")

                if not os.path.exists(stability_npy_path) or not os.path.exists(roi_npy_path):
                    print(f"     ⚠️ 找不到 {atlas_name} 的 .npy 文件，跳过此分区的绘图。")
                    continue

                # 直接极速加载计算好的结果
                stability_scores = np.load(stability_npy_path)
                nodal_strengths = np.load(roi_npy_path)

                # 【绘图 1】 Connectome (取最顶级的 1% 的边)
                for pct_label, pct_val in [('1pct', 0.99)]:
                    adj_matrix = build_adjacency_matrix(stability_scores, threshold_percentile=pct_val)
                    draw_connectome(
                        adjacency_matrix=adj_matrix,
                        atlas_img=atlas_nifti_path,
                        edge_threshold="90%",
                        save_dir=atlas_pics_dir,
                        # 【修改】：输出图片名称增加 modality
                        filename=f"Req1_Connectome_{model_group}_{modality}_{atlas_name}_top0.01.pdf"
                    )

                # 【绘图 2】 Stage 1 ROI (节点强度投影图)
                draw_atlas_roi(
                    roi_values=nodal_strengths,
                    atlas_img=atlas_nifti_path,
                    view="surf" if "schaefer" in atlas_name.lower() else "stat",
                    # 【修改】：标题和文件名增加 modality
                    title=f"[{modality}] {model_group.capitalize()} Intra-Atlas ROI ({atlas_name})",
                    save_dir=atlas_pics_dir,
                    filename=f"Req2_IntraROI_{model_group}_{modality}_{atlas_name}.pdf"
                )

            # -----------------------------------------------------------------
            # 需求 3 & 4: 跨分区全局一致性核心脑区 (GCI) 与 统计推断 (FWE)
            # -----------------------------------------------------------------
            print(f"\n>>> 正在生成 [Modality: {modality}] [{model_group.upper()}] 跨图谱融合与统计校正绘图...")
            
            # 【适配修改 3】 读取 CrossMethodAnal_Results 中的 NIfTI 对象
            stage_out_dir = os.path.join(BASE_DIR, 'CrossMethodAnal_Results', TASK_NAME)
            # 生成对应的合并图片的输出目录: results_CVCR/{DATASET}/CrossMethodAnal_Results/{TASK_NAME}/pics
            stage_pics_dir = os.path.join(stage_out_dir, 'pics')
            os.makedirs(stage_pics_dir, exist_ok=True)

            # 【修改】：文件名与 run_full_stages 保持一致，加上 modality 后缀
            gci_nifti_path = os.path.join(stage_out_dir, f'GCI_{DATASET}_{model_group}_{modality}_Merged_{len(ATLASES)}Atlases.nii.gz')
            fwe_nifti_path = os.path.join(stage_out_dir, f'GCI_{DATASET}_{model_group}_{modality}_FWECorrected.nii.gz')

            # 绘制 GCI 融合脑图
            if os.path.exists(gci_nifti_path):
                gci_nifti = nib.load(gci_nifti_path)
                display_gci = plot_stat_map(
                    gci_nifti,
                    # 【修改】：标题增加 modality
                    title=f"[{modality}] {model_group.capitalize()} Inter-Atlas GCI Map",
                    display_mode="ortho",
                    colorbar=True
                )
                _save_display(display_gci, os.path.join(stage_pics_dir, f"Req3_InterGCI_{model_group}_{modality}_Merged.pdf"))
            else:
                print(f"     ⚠️ 找不到 GCI 融合结果文件: {gci_nifti_path}")

            # 绘制 FWE 强统计推断核心脑图
            if os.path.exists(fwe_nifti_path):
                fwe_nifti = nib.load(fwe_nifti_path)
                display_fwe = plot_stat_map(
                    fwe_nifti,
                    # 【修改】：标题增加 modality
                    title=f"[{modality}] {model_group.capitalize()} FWE Corrected Core Regions (p<0.05)",
                    display_mode="ortho",
                    colorbar=True,
                    cmap='hot'
                )
                _save_display(display_fwe, os.path.join(stage_pics_dir, f"Req4_FWECorrected_{model_group}_{modality}_Merged.pdf"))
            else:
                print(f"     ⚠️ 找不到 FWE 校正结果文件: {fwe_nifti_path}")

            print(f"✅ [Modality: {modality}] [{model_group.upper()}] 分支所有可用可视化绘图已完成！")

if __name__ == "__main__":
    for dt in ['S1200', 'ABCD']:
        main(dataset_name=dt)