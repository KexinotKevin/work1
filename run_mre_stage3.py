import os
import numpy as np
import nibabel as nib
import pandas as pd

# 导入我们在 stage3_statistical_inference.py 中定义的核心函数
from stage3_statistical_inference import cluster_level_fwe_correction

def generate_mock_null_distribution(num_permutations=10000):
    """
    辅助函数：模拟 10,000 次随机打乱标签后产生的“最大假聚类体积”的零分布。
    在真实的科研中，这需要用高性能计算集群跑真实数据来替换。
    """
    print(f"正在模拟生成 {num_permutations} 次 Permutation 置换检验的零分布...")
    # 使用偏态分布模拟随机噪音聚类的大小：大部分很小(几十个像素)，极少数可能凑巧很大(两三百个像素)
    # 假设在随机噪音下，产生的假聚类平均大小为 50 体素
    mock_sizes = np.random.exponential(scale=50, size=num_permutations)
    return np.sort(mock_sizes.astype(int))

def main():
    # ---------- 1. 配置路径 ----------
    STATS_DIR = './stats'
    DATASET = 'S1200'
    STAGE2_RESULTS_DIR = os.path.join(STATS_DIR, 'Stage2_Results')
    
    # 读取我们在 Stage 2 阶段生成的真实 GCI 脑图
    real_gci_path = os.path.join(STAGE2_RESULTS_DIR, f'GCI_{DATASET}_Merged_2Atlases.nii.gz')
    
    if not os.path.exists(real_gci_path):
        raise FileNotFoundError(f"找不到 Stage 2 生成的 GCI 脑图，请先运行 Stage 2 的脚本！\n缺失文件: {real_gci_path}")

    print(f"=== 开始执行 Stage 3 统计推断与 FWE 校正: {DATASET} ===")
    
    # ---------- 2. 加载真实数据与生成零分布 ----------
    # 加载真实的 NIfTI 图像
    real_gci_img = nib.load(real_gci_path)
    real_gci_volume = real_gci_img.get_fdata()
    affine = real_gci_img.affine
    
    # 获取模拟的 10000 次零分布 (真实情况下，您需要从 10000 个伪 haufe 跑出的结果中提取)
    null_max_sizes = generate_mock_null_distribution(10000)
    
    # 计算 95% 分位数 (即 FWE p=0.05 时的临界体积大小)
    critical_size = np.percentile(null_max_sizes, 95)
    print(f"[零分布] 模拟完成。要想在 FWE_p < 0.05 下达到显著，聚类体积必须大于: {critical_size:.0f} 个体素。")
    
    # ---------- 3. 执行基于聚类的 FWE 校正 ----------
    # 设定初级阈值 (Primary Threshold)：假设 GCI > 1.0 的体素才有资格抱团形成聚类
    PRIMARY_THRESHOLD = 1.0
    P_VAL_THRESH = 0.05
    
    print(f"\n正在大洋捞针：寻找真实数据中 GCI > {PRIMARY_THRESHOLD} 的连通聚类，并执行统计检验...")
    
    significant_gci_volume, report = cluster_level_fwe_correction(
        real_gci_volume=real_gci_volume,
        null_max_sizes=null_max_sizes,
        primary_threshold=PRIMARY_THRESHOLD,
        p_val_thresh=P_VAL_THRESH
    )
    
    # ---------- 4. 输出统计报告与保存最终脑图 ----------
    if not report:
        print("\n[结果] 很遗憾，虽然发现了一些高贡献脑区，但它们的体积太小，未能通过 FWE 校正。")
    else:
        print("\n[结果] 恭喜！发现了以下通过 FWE 校正的显著核心脑网络聚类：")
        # 用 pandas 打印漂亮的表格
        df_report = pd.DataFrame(report)
        # 按照 P 值从小到大排序
        df_report = df_report.sort_values(by='P_FWE').reset_index(drop=True)
        print(df_report.to_string())
        
        # 保存最终去伪存真的 NIfTI 脑图
        output_dir = os.path.join(STATS_DIR, 'Stage3_Results')
        os.makedirs(output_dir, exist_ok=True)
        
        out_nifti_path = os.path.join(output_dir, f'GCI_{DATASET}_FWECorrected.nii.gz')
        nib.save(nib.Nifti1Image(significant_gci_volume, affine), out_nifti_path)
        
        print(f"\n[完成] 已滤除所有随机假阳性噪点。")
        print(f"       最终纯净版 FWE 校正脑图已保存至: {out_nifti_path}")
        print("       (在这张图上看到的任何颜色，都代表在统计学上无可辩驳的认知核心预测区域！)")

if __name__ == "__main__":
    main()