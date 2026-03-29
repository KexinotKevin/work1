import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def load_model_weights(stats_dir, task, dataset, atlas, model_group='linear'):
    """
    【最新修正版】：统一读取 haufe 权重文件，并通过文件名中的 method 字段进行精确分组过滤。
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
        
    # 统一抓取所有 haufe 文件
    search_pattern = os.path.join(stats_dir, task, f'haufe__dataset_{dataset}__atlas_{atlas}__*.csv')
    all_file_paths = glob.glob(search_pattern)
    
    weights_list = []
    file_names = []
    
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
            
    if not file_names:
        print(f"   (提示: 当前配置下未找到属于 [{model_group}] 组的有效文件)")
        
    return np.array(weights_list), file_names

def percentile_rank_transform(weight_matrix):
    """
    第一步：贡献度量纲标准化与秩转换
    将每个模态/模型的原始权重转化为 (0, 1] 的百分位秩。
    
    参数:
        weight_matrix (np.ndarray): 形状为 (K, D) 的矩阵。K 为模态数，D 为边数。
        
    返回:
        rank_matrix (np.ndarray): 转换后的秩矩阵，形状同上。
    """
    K, D = weight_matrix.shape
    rank_matrix = np.zeros_like(weight_matrix, dtype=np.float64)
    
    for k in range(K):
        # 注意：Haufe 权重反映的是协方差/贡献度，绝对值越大代表贡献越强
        # 因此我们先取绝对值，再计算秩次 (默认升序，最大的绝对值秩次最接近 D)
        abs_weights = np.abs(weight_matrix[k, :])
        
        # rankdata 分配从 1 到 D 的秩次，除以 D 映射到 (0, 1] 范围
        rank_matrix[k, :] = rankdata(abs_weights) / D
        
    return rank_matrix

def calculate_stability_score(rank_matrix):
    """
    第二步：跨模态稳定得分（Stability Score）的度量
    计算每条边在所有模态/模型下的几何平均得分。
    
    参数:
        rank_matrix (np.ndarray): 形状为 (K, D) 的秩矩阵。
        
    返回:
        stability_scores (np.ndarray): 形状为 (D,) 的一维数组，代表每条边的稳定性得分。
    """
    # 数学说明：直接连乘 k 个小数可能导致数值下溢（浮点数精度丢失）。
    # 稳健的计算方式是利用对数转换：(x1*x2*...*xk)^(1/k) = exp( mean( ln(x1), ln(x2), ..., ln(xk) ) )
    # 因为秩的范围是 (0, 1]，不用担心 log 负数问题。
    
    log_ranks = np.log(rank_matrix)
    stability_scores = np.exp(np.mean(log_ranks, axis=0))
    return stability_scores

def edge_to_node_mapping(stability_scores, threshold_percentile=None):
    """
    第三步：节点级预测贡献度映射（Nodal Mapping）
    基于加权节点强度理论，将边贡献映射至节点。
    
    参数:
        stability_scores (np.ndarray): 形状为 (D,) 的边得分数组。
        threshold_percentile (float, 可选): 例如 0.90，表示仅聚合前 10% 的显著边。如为 None 则聚合所有边。
        
    返回:
        nodal_strengths (np.ndarray): 形状为 (N,) 的节点得分数组。
        N (int): 节点数量。
    """
    D = len(stability_scores)
    # 反推节点数 N：已知边数 D = N*(N-1)/2，解一元二次方程 N^2 - N - 2D = 0
    N = int(np.round((1 + np.sqrt(1 + 8 * D)) / 2))
    
    # 构建 N x N 对称连接矩阵
    adj_matrix = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    
    # 显著性边筛选：论文中提到 E_sig。这里提供一种简单的百分位阈值筛选法
    if threshold_percentile is not None:
        threshold_val = np.quantile(stability_scores, threshold_percentile)
        # 将低于阈值的得分置为 0，不参与节点聚合
        filtered_scores = np.where(stability_scores >= threshold_val, stability_scores, 0.0)
    else:
        filtered_scores = stability_scores
        
    # 填充上三角并对称化
    adj_matrix[iu] = filtered_scores
    adj_matrix = adj_matrix + adj_matrix.T
    
    # 节点度量：沿着相邻节点求和（对应公式中 \sum_{j \in \mathcal{N}(v)}）
    nodal_strengths = np.sum(adj_matrix, axis=1)
    
    return nodal_strengths, N