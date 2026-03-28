import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

# 导入您仓库中现成的绘图模块
from draw_brain import draw_connectome, draw_atlas_roi, _save_display

# 导入我们这几轮对话中编写的 1~3 阶段核心计算函数
from stage1_intra_atlas import (
    load_model_weights, 
    percentile_rank_transform, 
    calculate_stability_score, 
    edge_to_node_mapping
)
from stage2_inter_atlas import (
    load_and_align_atlas, 
    project_to_voxel_space, 
    calculate_gci_and_confidence
)
from stage3_statistical_inference import cluster_level_fwe_correction

def build_adjacency_matrix(stability_scores, threshold_percentile):
    """辅助函数：将一维得分重建为 NxN 邻接矩阵，并过滤阈值"""
    D = len(stability_scores)
    N = int(np.round((1 + np.sqrt(1 + 8 * D)) / 2))
    threshold_val = np.quantile(stability_scores, threshold_percentile)
    filtered_scores = np.where(stability_scores >= threshold_val, stability_scores, 0.0)
    
    adj_matrix = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    adj_matrix[iu] = filtered_scores
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def mock_null_distribution(num_permutations=10000):
    """模拟一个 10000 次置换检验的零分布，以供 Stage 3 绘图演示跑通"""
    return np.sort(np.random.exponential(scale=50, size=num_permutations).astype(int))

def main(task_name=None):
    # ---------- 全局配置 ----------
    DATASET = 'S1200'
    STATS_DIR = f'./res_{DATASET}/stats'
    PICS_DIR = './pics/Stage_Results_Vis'
    TASK_NAME = 'CogFluidComp_Unadj' if task_name is None else task_name
    ATLASES = ['bna246', 'schaefer200_S1']
    ATLAS_DIR = "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"
    
    # 按照您的要求，线性和非线性分开独立处理
    MODEL_TYPES = ['linear', 'nonlinear']
    
    os.makedirs(PICS_DIR, exist_ok=True)
    
    for model_type in MODEL_TYPES:
        print(f"\n{'='*50}")
        print(f"🚀 开始处理模型流: [{model_type.upper()}] (特征溯源: {'Haufe' if model_type=='linear' else 'SHAP'})")
        print(f"{'='*50}")
        
        all_gci_volumes = []
        reference_affine = None
        reference_atlas_path = None
        
        # -----------------------------------------------------------------
        # 需求 1 & 2: 同种分区、跨方法的关键连边 (Connectome) 与脑区 (ROI)
        # -----------------------------------------------------------------
        for atlas_name in ATLASES:
            print(f"\n>>> 正在分析分区: {atlas_name}")
            atlas_nifti_path = os.path.join(ATLAS_DIR, f"{atlas_name}.nii.gz")
            
            # 设置空间基准
            if reference_atlas_path is None:
                reference_atlas_path = atlas_nifti_path
            aligned_atlas_img, _ = load_and_align_atlas(atlas_nifti_path, reference_atlas_path)
            if reference_affine is None:
                reference_affine = aligned_atlas_img.affine
            
            # [Stage 1 计算]
            weight_matrix, file_names = load_model_weights(STATS_DIR, TASK_NAME, DATASET, atlas_name, model_type=model_type)
            if len(weight_matrix) == 0:
                continue
                
            rank_matrix = percentile_rank_transform(weight_matrix)
            stability_scores = calculate_stability_score(rank_matrix)
            
            # 【需求 1】生成并保存连边图 (分别保留前 10%, 30%, 50%)
            # 对应的百分位阈值分别是 0.90, 0.70, 0.50
            for pct_label, pct_val in [('10pct', 0.90), ('30pct', 0.70), ('50pct', 0.50)]:
                adj_matrix = build_adjacency_matrix(stability_scores, threshold_percentile=pct_val)
                draw_connectome(
                    adjacency_matrix=adj_matrix,
                    atlas_img=atlas_nifti_path,
                    edge_threshold="0%", # 已在矩阵重建时过滤，此处不再过滤
                    title=f"{model_type.capitalize()} Connectome (Top {pct_label}) - {atlas_name}",
                    save_dir=PICS_DIR,
                    filename=f"Req1_Connectome_{model_type}_{atlas_name}_top{pct_label}.png"
                )
            
            # 【需求 2】生成并保存同分区下的关键脑区分布图 (Stage 1 ROI)
            # 脑区强度计算我们统一定义为聚合前 5% 的强连接
            nodal_strengths, _ = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)
            draw_atlas_roi(
                roi_values=nodal_strengths,
                atlas_img=atlas_nifti_path,
                view="surf" if "schaefer" in atlas_name else "stat", # Schaefer 优先用皮层渲染
                title=f"{model_type.capitalize()} Intra-Atlas ROI ({atlas_name})",
                save_dir=PICS_DIR,
                filename=f"Req2_IntraROI_{model_type}_{atlas_name}.png"
            )
            
            # [为 Stage 2 准备投影数据]
            volume_data, _ = project_to_voxel_space(nodal_strengths, aligned_atlas_img)
            all_gci_volumes.append(volume_data)

        # 确保该模型类型下有提取到数据再进行融合
        if not all_gci_volumes:
            continue

        # -----------------------------------------------------------------
        # 需求 3: 跨分区、跨方法的全局一致性关键脑区 (读取 Stage 2 保存的 NIfTI)
        # -----------------------------------------------------------------
        print(f"\n>>> 正在加载并绘制 {model_type.upper()} 跨图谱融合 (Stage 2 GCI)...")
        # 【修改点】：这里直接拼凑您刚才在流水线里设定的新文件名
        stage2_file = os.path.join(STATS_DIR, 'Stage2_Results', f'GCI_{DATASET}_{model_type}_Merged_{len(ATLASES)}Atlases.nii.gz')
        
        if os.path.exists(stage2_file):
            gci_nifti = nib.load(stage2_file)
            display_gci = plot_stat_map(
                gci_nifti, 
                title=f"{model_type.capitalize()} Inter-Atlas GCI Map",
                display_mode="ortho",
                colorbar=True
            )
            _save_display(display_gci, os.path.join(PICS_DIR, f"Req3_InterGCI_{model_type}_Merged.png"))
        else:
            print(f"⚠️ 找不到 Stage 2 文件，跳过绘图: {stage2_file}")

        # -----------------------------------------------------------------
        # 需求 4: 经过统计校正后的严谨解剖核心脑区 (读取 Stage 3 保存的 NIfTI)
        # -----------------------------------------------------------------
        print(f"\n>>> 正在加载并绘制 {model_type.upper()} 统计推断与 FWE 校正 (Stage 3)...")
        # 【修改点】：直接读取带有 model_type 标签的 FWE 结果
        stage3_file = os.path.join(STATS_DIR, 'Stage3_Results', f'GCI_{DATASET}_{model_type}_FWECorrected.nii.gz')
        
        if os.path.exists(stage3_file):
            fwe_nifti = nib.load(stage3_file)
            display_fwe = plot_stat_map(
                fwe_nifti,
                title=f"{model_type.capitalize()} FWE Corrected Core Regions (p<0.05)",
                display_mode="ortho",
                colorbar=True,
                cmap='hot' 
            )
            _save_display(display_fwe, os.path.join(PICS_DIR, f"Req4_FWECorrected_{model_type}_Merged.png"))
        else:
            print(f"⚠️ 找不到 Stage 3 文件，跳过绘图: {stage3_file}")
if __name__ == "__main__":
    main()