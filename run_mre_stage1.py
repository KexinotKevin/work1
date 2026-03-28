import os
import numpy as np
from stage1_intra_atlas import (
    load_haufe_weights,
    percentile_rank_transform,
    calculate_stability_score,
    edge_to_node_mapping
)

def main():
    # 1. 配置参数
    STATS_DIR = './stats'
    TASK_NAME = 'CogFluidComp_Unadj'  # 以流体智力未校正结果为例
    DATASET = 'S1200'
    ATLAS = 'bna246'
    
    print(f"=== 开始运行第一阶段分析: {DATASET} | {ATLAS} ===")
    
    try:
        # Step 1: 读取所有相关的预测模型贡献度（跨模态、跨方法）
        weight_matrix, file_names = load_haufe_weights(
            stats_dir=STATS_DIR,
            task=TASK_NAME,
            dataset=DATASET,
            atlas=ATLAS
        )
        print(f"成功加载 {len(file_names)} 个模型的权重文件。")
        print(f"特征维度 (连接边数量): {weight_matrix.shape[1]}")
        
        # Step 2: 量纲标准化与秩转换
        rank_matrix = percentile_rank_transform(weight_matrix)
        print(f"完成百分位秩转换，秩矩阵形状: {rank_matrix.shape}")
        
        # Step 3: 计算跨模态几何平均稳定得分
        stability_scores = calculate_stability_score(rank_matrix)
        print(f"完成稳定得分计算。最高得分: {stability_scores.max():.4f}, 最低得分: {stability_scores.min():.4f}")
        
        # Step 4: 边到节点的映射 (这里以仅聚合排名前 5% 的边为例，模拟论文中的 E_sig)
        # BNA 分区标准包含 246 个脑区，因此最终的 N 应为 246
        nodal_strengths, num_nodes = edge_to_node_mapping(stability_scores, threshold_percentile=0.95)
        print(f"完成节点映射，反推图谱节点数 N = {num_nodes}")
        
        # 结果输出展示
        print("\n--- Top 5 贡献度最高的脑区节点 (Node ID: Strength) ---")
        top_node_indices = np.argsort(nodal_strengths)[::-1][:5]
        for idx in top_node_indices:
            print(f"节点 ID: {idx + 1:3d} (索引 {idx:3d}), 节点强度: {nodal_strengths[idx]:.4f}")
            
        # (可选) 将节点得分保存到磁盘，供第二阶段空间投影使用
        output_dir = os.path.join(STATS_DIR, 'Stage1_Results')
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'nodal_strength_{DATASET}_{ATLAS}.npy')
        np.save(out_file, nodal_strengths)
        print(f"\n节点得分已保存至: {out_file}")

    except Exception as e:
        print(f"\n运行中断: {e}")
        print("提示: 请确认 './stats' 目录下是否包含 bna246 图谱对于 S1200 数据集的 haufe CSV 文件。")

if __name__ == "__main__":
    main()