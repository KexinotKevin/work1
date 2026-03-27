import numpy as np
import argparse
try:
    import shap
except ImportError:  # pragma: no cover - optional dependency for SHAP mode
    shap = None
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

# 直接从你的 edge_selection.py 中导入所需函数
from edge_selection import _infer_num_nodes, rearrange_edges, reshape_feat_to_net


def get_edge_contributions_symmetric(model, X_data, y_data, method='shap'):
    """
    计算并返回 N x N 的【严格对称】边贡献度矩阵。
    X_data 形状应为 (N_samples, num_edges)，即仅包含上三角特征的数据。
    """
    num_edges = X_data.shape[1]
    num_nodes = _infer_num_nodes(num_edges)
    
    if method == 'shap':
        if shap is None:
            raise ImportError("SHAP is not installed. Use method='permutation' or install shap.")
        model_type = type(model).__name__
        if model_type in ['DecisionTreeRegressor', 'GradientBoostingRegressor']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data)
        else:
            # 对于非树模型，使用 KMeans 提取背景集以加速 KernelExplainer
            background = shap.kmeans(X_data, 10) 
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_data, nsamples=50)
            
        # 计算全局重要度：对所有样本的 SHAP 绝对值求平均
        global_importances = np.abs(shap_values).mean(axis=0)
        
    elif method == 'permutation':
        result = permutation_importance(
            model, X_data, y_data, n_repeats=5, random_state=42, n_jobs=1
        )
        global_importances = result.importances_mean
        
    else:
        raise ValueError("不支持的解释方法，请选择 'shap' 或 'permutation'")
        
    # 直接调用你的函数：将一维的特征重要性重塑为对角线为0的对称矩阵
    symmetric_contribution_matrix = reshape_feat_to_net(global_importances, num_nodes)
    
    return symmetric_contribution_matrix


# ==========================================
# 最小可运行测试 (MRE) 执行模块
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["shap", "permutation"],
        default="permutation",
        help="Feature attribution method.",
    )
    args = parser.parse_args()

    # 使用较小的节点数和样本数以便快速验证
    N_samples = 100
    M_nodes = 10 
    
    print(f"正在生成模拟数据 (样本数={N_samples}, 节点数={M_nodes})...")
    
    # 1. 模拟生成输入数据 (M, M, N)
    mock_graphs = np.zeros((M_nodes, M_nodes, N_samples))
    for i in range(N_samples):
        # 随机生成对称邻接矩阵，并对角线置 0
        mat = np.random.rand(M_nodes, M_nodes)
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)
        mock_graphs[:, :, i] = mat
        
    # 模拟生成标签数据 (N,)
    y_train = np.random.rand(N_samples) * 10 
    
    # 2. 提取特征，准备训练
    # 直接调用 rearrange_edges，提取上三角边
    train_picked_edges = rearrange_edges(mock_graphs, M_nodes, N_samples)
    
    # rearrange_edges 返回的是 (num_edges, N_samples)，训练前需要转置
    X_train = train_picked_edges.T 
    print(f"特征提取完毕，X_train 形状: {X_train.shape} (样本数, 边数)")
    
    # 3. 训练非线性模型
    print("\n正在训练 Gradient Boosting 模型...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. 计算并提取对称的解释性矩阵
    print("正在计算 SHAP 贡献度矩阵...")
    shap_matrix = get_edge_contributions_symmetric(model, X_train, y_train, method=args.method)
    
    # 5. 验证结果
    print("\n=== 验证结果 ===")
    print(f"SHAP 矩阵形状: {shap_matrix.shape}")
    print(f"对角线是否全为 0: {np.all(np.diag(shap_matrix) == 0)}")
    
    # 检查矩阵是否严格对称 (矩阵 == 其转置矩阵)
    is_symmetric = np.allclose(shap_matrix, shap_matrix.T)
    print(f"重构后的矩阵是否严格对称: {is_symmetric}")
    
    print("\nSHAP 贡献度矩阵局部示例 (前 5x5):")
    np.set_printoptions(precision=4, suppress=True)
    print(shap_matrix[:5, :5])
