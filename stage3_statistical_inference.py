import numpy as np
from scipy.ndimage import label

def find_clusters(volume, primary_threshold):
    """
    寻找 3D 空间中超过初级阈值的相连体素聚类（Clusters）。
    
    参数:
        volume (np.ndarray): 3D 脑图矩阵 (如 GCI 地图)
        primary_threshold (float): 初级体素形成阈值 (例如 Z > 3.1)
        
    返回:
        cluster_labels (np.ndarray): 标记了不同聚类的 3D 矩阵 (0 为背景，1,2,3... 为不同聚类)
        cluster_sizes (list): 每个聚类包含的体素数量
    """
    # 1. 对脑图进行二值化：只保留大于初级阈值的体素
    binary_map = volume > primary_threshold
    
    # 2. 定义 3D 空间的连通性 (26连通，即对角线相连也算同一个聚类)
    structure = np.ones((3, 3, 3), dtype=int)
    
    # 3. 寻找连通区域
    cluster_labels, num_clusters = label(binary_map, structure=structure)
    
    # 4. 计算每个聚类的大小
    cluster_sizes = []
    for i in range(1, num_clusters + 1):
        size = np.sum(cluster_labels == i)
        cluster_sizes.append(size)
        
    return cluster_labels, cluster_sizes

def build_null_distribution_of_max_clusters(null_gci_volumes, primary_threshold):
    """
    从 10000 张由于打乱标签产生的假 GCI 地图中，构建最大聚类大小的零分布。
    """
    max_cluster_sizes_null = []
    
    for null_vol in null_gci_volumes:
        _, sizes = find_clusters(null_vol, primary_threshold)
        if sizes:
            # 记录这张假脑图里“最大的那个由随机噪音产生的斑块”有多大
            max_cluster_sizes_null.append(max(sizes))
        else:
            max_cluster_sizes_null.append(0)
            
    # 从小到大排序，方便后续计算 P 值
    return np.sort(max_cluster_sizes_null)

def cluster_level_fwe_correction(real_gci_volume, null_max_sizes, primary_threshold, p_val_thresh=0.05):
    """
    执行基于聚类的多重比较校正 (Cluster-level FWE Correction)。
    
    参数:
        real_gci_volume: 真实的 3D GCI 地图
        null_max_sizes: 10000 次置换检验得到的零分布数组
        primary_threshold: 形成聚类的初级阈值
        p_val_thresh: 最终的族系误差率 (FWE) 显著性水平，通常为 0.05
        
    返回:
        significant_gci_volume: 经过 FWE 校正后，只保留显著聚类的 3D 地图
        report: 各个显著聚类的统计信息
    """
    num_permutations = len(null_max_sizes)
    
    # 1. 寻找真实数据中的聚类
    real_labels, real_sizes = find_clusters(real_gci_volume, primary_threshold)
    
    significant_gci_volume = np.zeros_like(real_gci_volume)
    report = []
    
    # 2. 检验真实数据中的每一个聚类是否显著
    for i, size in enumerate(real_sizes):
        cluster_id = i + 1
        
        # 计算非参数 P 值：在 10000 次随机打乱中，有多少次产生的最大假聚类，比我们现在这个真实聚类还要大？
        # 如果这个概率小于 0.05，说明我们这个真实聚类不可能是随机噪音产生的。
        count_larger_in_null = np.sum(null_max_sizes >= size)
        p_fwe = count_larger_in_null / num_permutations
        
        if p_fwe < p_val_thresh:
            # 只有显著的聚类，才被填入最终的校正地图中
            mask = real_labels == cluster_id
            significant_gci_volume[mask] = real_gci_volume[mask]
            
            report.append({
                'Cluster_ID': cluster_id,
                'Voxel_Count': size,
                'Max_GCI': np.max(real_gci_volume[mask]),
                'P_FWE': p_fwe
            })
            
    return significant_gci_volume, report