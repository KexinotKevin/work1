import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def load_model_weights(stats_dir, task, dataset, atlas, model_group='linear'):
    """
    【最新修正版】：统一读取 Haufe/permutation 权重文件，并通过文件名中的 method 字段进行精确分组过滤。

    目录结构:
    - res_{dataset}/{atlas}/stats/{task}/haufe__label_{label}__dataset_*__atlas_*__modality_*__type_*.csv  (线性模型)
    - res_{dataset}/{atlas}/permutation/{task}/permutation__label_{label}__dataset_*__atlas_*__modality_*__type_*.csv  (非线性模型)
    """
    # 定义严格的算法分组映射表
    group_mapping = {
        'linear': ['lasso', 'ridge', 'linear', 'huber'],
        'nonlinear': ['kernel_ridge_rbf', 'decision_tree', 'gradient_boosting'],
        'mlp': ['single_layer_mlp']
    }

    allowed_methods = group_mapping.get(model_group, [])
    if not allowed_methods:
        print(f"⚠️ 未知的模型分组: {model_group}")
        return np.array([]), []

    # 新的目录结构: res_{dataset}/{atlas}/{type}/{task}/
    atlas_dir = os.path.join(stats_dir, "..")  # 回到 res_{dataset}/{atlas} 目录

    weights_list = []
    file_names = []

    # 线性模型使用 haufe 文件（在 stats 目录下）
    # 匹配: haufe__label_{task}__dataset_*__atlas_{atlas}__modality_*__type_*.csv
    if model_group == 'linear':
        # 【修改这里】：去掉 os.path.join 中多余的 task 层级
        search_pattern = os.path.join(
            stats_dir,
            f'haufe__label_{task}__dataset_*__atlas_{atlas}__modality_*__type_*__method_*.csv'
        )
        all_file_paths = glob.glob(search_pattern)

        for fpath in all_file_paths:
            fname = os.path.basename(fpath)

            # 精确匹配 method 字段：检查文件名中是否包含 "__method_算法名.csv"
            is_match = any(f"__method_{m}.csv" in fname for m in allowed_methods)

            if is_match:
                df = pd.read_csv(fpath)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])

                vals = df.values
                if vals.ndim == 2 and vals.shape[0] == vals.shape[1]:
                    iu = np.triu_indices(vals.shape[0], k=1)
                    vec = vals[iu]
                else:
                    vec = vals.flatten()

                weights_list.append(vec)
                file_names.append(fname)

    # 非线性模型使用 permutation 文件（在 permutation 目录下）
    # 匹配: permutation__label_{task}__dataset_*__atlas_{atlas}__modality_*__type_*.csv
    elif model_group in ('nonlinear', 'mlp'):
        # 【修改这里】：去掉 os.path.join 中多余的 task 层级
        permutation_dir = os.path.join(stats_dir, "..", "permutation")
        search_pattern = os.path.join(
            permutation_dir,
            f'permutation__label_{task}__dataset_*__atlas_{atlas}__modality_*__type_*__method_*.csv'
        )
        all_file_paths = glob.glob(search_pattern)

        for fpath in all_file_paths:
            fname = os.path.basename(fpath)

            # 精确匹配 method 字段
            is_match = any(f"__method_{m}.csv" in fname for m in allowed_methods)

            if is_match:
                df = pd.read_csv(fpath)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])

                vals = df.values
                if vals.ndim == 2 and vals.shape[0] == vals.shape[1]:
                    iu = np.triu_indices(vals.shape[0], k=1)
                    vec = vals[iu]
                else:
                    vec = vals.flatten()

                weights_list.append(vec)
                file_names.append(fname)

    if not file_names:
        print(f"   (提示: 当前配置下未找到属于 [{model_group}] 组的有效文件)")

    return np.array(weights_list), file_names

def percentile_rank_transform(weight_matrix):
    """
    第一步：贡献度量纲标准化与秩转换（保留符号信息）
    将每个模态/模型的原始权重转化为有符号的百分位秩。
    - 正值映射到 (0.5, 1] 范围
    - 负值映射到 [-1, -0.5) 范围
    - 零值保持为 0
    
    参数:
        weight_matrix (np.ndarray): 形状为 (K, D) 的矩阵。K 为模态数，D 为边数。
        
    返回:
        rank_matrix (np.ndarray): 转换后的有符号秩矩阵，形状同上。
    """
    K, D = weight_matrix.shape
    rank_matrix = np.zeros_like(weight_matrix, dtype=np.float64)
    
    for k in range(K):
        weights_k = weight_matrix[k, :]
        pos_mask = weights_k > 0
        neg_mask = weights_k < 0
        
        # 正值的百分位秩：映射到 (0.5, 1]
        if np.any(pos_mask):
            pos_abs = np.abs(weights_k[pos_mask])
            if pos_abs.max() > pos_abs.min():
                pos_ranks = rankdata(pos_abs) / len(pos_abs)
            else:
                pos_ranks = np.ones(len(pos_abs))
            # 映射到 [0.5, 1]，0 表示无贡献
            rank_matrix[k, pos_mask] = 0.5 + 0.5 * pos_ranks
        
        # 负值的百分位秩：映射到 [-1, -0.5)
        if np.any(neg_mask):
            neg_abs = np.abs(weights_k[neg_mask])
            if neg_abs.max() > neg_abs.min():
                neg_ranks = rankdata(neg_abs) / len(neg_abs)
            else:
                neg_ranks = np.ones(len(neg_abs))
            # 映射到 [-1, -0.5]
            rank_matrix[k, neg_mask] = -0.5 - 0.5 * neg_ranks
        
        # 零值保持为 0
    
    return rank_matrix

def calculate_stability_score(rank_matrix):
    """
    第二步：跨模态稳定得分（Stability Score）的度量（保留符号）
    计算每条边在所有模态/模型下的几何平均得分，保留正负符号。
    
    计算逻辑：
    1. 分别对正负边计算几何平均强度
    2. 基于投票机制决定最终符号
    3. 如果多数模型认为是正向贡献则为正，否则为负
    
    参数:
        rank_matrix (np.ndarray): 形状为 (K, D) 的有符号秩矩阵。
        
    返回:
        stability_scores (np.ndarray): 形状为 (D,) 的一维数组，
                                        代表每条边的稳定性得分。
                                        正值表示正相关稳定，负值表示负相关稳定。
    """
    K, D = rank_matrix.shape
    
    # 获取每条边被标记为正/负的模型数量
    pos_votes = np.sum(rank_matrix > 0, axis=0)  # 正值投票数
    neg_votes = np.sum(rank_matrix < 0, axis=0)  # 负值投票数
    total_votes = pos_votes + neg_votes  # 非零投票数
    
    # 计算正值的几何平均
    pos_abs_mean = np.zeros(D)
    pos_mask = pos_votes > 0
    if np.any(pos_mask):
        pos_rank_abs = np.abs(rank_matrix[:, pos_mask])
        log_pos = np.log(pos_rank_abs + 1e-10)
        pos_abs_mean[pos_mask] = np.exp(np.mean(log_pos, axis=0))
    
    # 计算负值的几何平均（取绝对值）
    neg_abs_mean = np.zeros(D)
    neg_mask = neg_votes > 0
    if np.any(neg_mask):
        neg_rank_abs = np.abs(rank_matrix[:, neg_mask])
        log_neg = np.log(neg_rank_abs + 1e-10)
        neg_abs_mean[neg_mask] = np.exp(np.mean(log_neg, axis=0))
    
    # 基于投票决定最终符号和强度
    stability_scores = np.zeros(D)
    
    # 多数票胜出
    # 如果正票 > 负票，使用正值强度；否则使用负值强度
    pos_wins = pos_votes > neg_votes
    neg_wins = neg_votes > pos_votes
    tie = (pos_votes == neg_votes) & (total_votes > 0)
    
    stability_scores[pos_wins] = pos_abs_mean[pos_wins]
    stability_scores[neg_wins] = -neg_abs_mean[neg_wins]
    
    # 平票时：如果有非零票，取绝对值较大者的符号
    # 如果无任何票（全是零），保持为 0
    if np.any(tie):
        tie_pos = pos_abs_mean[tie]
        tie_neg = neg_abs_mean[tie]
        tie_sign = np.where(tie_pos >= tie_neg, 1, -1)
        tie_abs = np.maximum(tie_pos, tie_neg)
        stability_scores[tie] = tie_sign * tie_abs
    
    return stability_scores

def edge_to_node_mapping(stability_scores, threshold_percentile=None):
    """
    第三步：节点级预测贡献度映射（Nodal Mapping）
    基于加权节点强度理论，将边贡献映射至节点。
    
    重要：本函数现在支持有符号的 stability_scores：
    - 正值表示正向贡献（正相关），负值表示负向贡献（负相关）
    - 阈值筛选基于绝对值，确保正负显著边都被考虑
    - 节点强度反映该节点参与的所有显著边的代数和
    
    参数:
        stability_scores (np.ndarray): 形状为 (D,) 的有符号边得分数组。
        threshold_percentile (float, 可选): 例如 0.90，表示仅聚合绝对值前 10% 的显著边。
                                           如为 None 则聚合所有边。
        
    返回:
        nodal_strengths (np.ndarray): 形状为 (N,) 的节点得分数组。
                                      正值表示正向贡献为主的节点，负值表示负向贡献为主的节点。
        N (int): 节点数量。
    """
    D = len(stability_scores)
    # 反推节点数 N：已知边数 D = N*(N-1)/2，解一元二次方程 N^2 - N - 2D = 0
    N = int(np.round((1 + np.sqrt(1 + 8 * D)) / 2))
    
    # 构建 N x N 对称连接矩阵（保留符号）
    adj_matrix = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    
    # 显著性边筛选：基于绝对值进行百分位阈值筛选
    if threshold_percentile is not None:
        abs_scores = np.abs(stability_scores)
        threshold_val = np.quantile(abs_scores, threshold_percentile)
        # 保留符号，只保留绝对值大于阈值的边
        filtered_scores = np.where(abs_scores >= threshold_val, stability_scores, 0.0)
    else:
        filtered_scores = stability_scores
    
    # 填充上三角并对称化
    adj_matrix[iu] = filtered_scores
    adj_matrix = adj_matrix + adj_matrix.T
    
    # 节点度量：沿着相邻节点求和（对应公式中 \sum_{j \in \mathcal{N}(v)}）
    # 正负连边会相互抵消，反映该节点的"净"连接特性
    nodal_strengths = np.sum(adj_matrix, axis=1)
    
    return nodal_strengths, N


def edge_to_node_mapping_signed(stability_scores, threshold_percentile=None):
    """
    第三步（备选版本）：节点级预测贡献度映射（分别统计正负贡献）
    
    与 edge_to_node_mapping 不同，本函数分别计算正向和负向节点贡献，
    返回两个独立的节点强度数组。
    
    参数:
        stability_scores (np.ndarray): 形状为 (D,) 的有符号边得分数组。
        threshold_percentile (float, 可选): 例如 0.90，表示仅聚合绝对值前 10% 的显著边。
        
    返回:
        nodal_strengths_pos (np.ndarray): 形状为 (N,) 的正向节点强度数组。
        nodal_strengths_neg (np.ndarray): 形状为 (N,) 的负向节点强度数组。
        N (int): 节点数量。
    """
    D = len(stability_scores)
    N = int(np.round((1 + np.sqrt(1 + 8 * D)) / 2))
    
    # 分离正负连边
    pos_scores = np.maximum(stability_scores, 0.0)  # 只保留正值
    neg_scores = np.minimum(stability_scores, 0.0)  # 只保留负值
    
    # 阈值筛选（基于原始绝对值）
    if threshold_percentile is not None:
        abs_scores = np.abs(stability_scores)
        threshold_val = np.quantile(abs_scores, threshold_percentile)
        pos_scores = np.where(abs_scores >= threshold_val, pos_scores, 0.0)
        neg_scores = np.where(abs_scores >= threshold_val, neg_scores, 0.0)
    
    # 构建正负邻接矩阵
    adj_pos = np.zeros((N, N))
    adj_neg = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    
    adj_pos[iu] = pos_scores
    adj_pos = adj_pos + adj_pos.T
    
    adj_neg[iu] = np.abs(neg_scores)  # 负值取绝对值
    adj_neg = adj_neg + adj_neg.T
    
    # 节点强度 = 入射边的加权和
    nodal_strengths_pos = np.sum(adj_pos, axis=1)
    nodal_strengths_neg = np.sum(adj_neg, axis=1)
    
    return nodal_strengths_pos, nodal_strengths_neg, N