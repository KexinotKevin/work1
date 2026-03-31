"""
去交杂功能测试脚本

在ABCD数据集上测试DeconfoundWrapper的效果：
- 使用linear方案（lasso, ridge, linear, huber）
- 包含全部modal（SC + FC含正负连接）
- 对比有/无去交杂的结果差异
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from datasets_base import (
    FC_KIND, SC_KIND, SFC_KIND, load_data, normalize_modal_type,
    FC_CONN_MASK
)
from datasets_cfg import atlases, get_dataset_cfg
from edge_selection import select_sig_edges
from models import get_regression_model, DeconfoundWrapper


# Linear methods to test
LINEAR_METHODS = [
    {"method_type": "linear", "method_name": "lasso", "params": {"alpha": 0.05, "max_iter": 5000}},
    {"method_type": "linear", "method_name": "ridge", "params": {"alpha": 1.0}},
    {"method_type": "linear", "method_name": "linear", "params": {}},
    {"method_type": "linear", "method_name": "huber", "params": {"epsilon": 1.35, "max_iter": 1000}},
]


def evaluate_predictions(y_true, y_pred):
    """评估预测效果"""
    if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        pred_r, pred_p = pearsonr(y_true, y_pred)
    else:
        pred_r, pred_p = np.nan, np.nan
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pred_r) if not np.isnan(pred_r) else np.nan,
        "pearson_p": float(pred_p) if not np.isnan(pred_p) else np.nan,
    }


def expand_fc_conn_kinds(conn_kinds):
    """将FC conn_kinds展开为原始+正连接+负连接"""
    result = []
    for k in conn_kinds:
        result.append(k)
        pos_key = f"{k}_pos"
        neg_key = f"{k}_neg"
        if pos_key in FC_CONN_MASK and neg_key in FC_CONN_MASK:
            result.append(pos_key)
            result.append(neg_key)
    return result


def run_single_experiment(
    dataset_name,
    atlas_name,
    modal_type,
    conn_kind,
    label_name,
    use_deconfound=True,
    random_state=42,
    n_splits=10,
):
    """对单个实验组合运行测试"""
    print(f"\n{'='*60}")
    print(f"[{'去交杂' if use_deconfound else '无去交杂'}] "
          f"dataset={dataset_name} atlas={atlas_name} "
          f"modal={modal_type} conn_kind={conn_kind} label={label_name}")
    print('='*60)

    # 加载数据
    dt_cfg = get_dataset_cfg(dataset_name)
    dataset = load_data(
        dataset_name=dataset_name,
        dt_cfg=dt_cfg,
        atlas_name=atlas_name,
        modal_type=modal_type,
        sub_kind=conn_kind,
    )

    mats = dataset["nets"][atlas_name][conn_kind]
    scores_df = dataset["scores"].copy()

    # 提取标签和协变量
    label_values = pd.to_numeric(scores_df[label_name], errors="coerce").to_numpy()
    valid_mask = np.isfinite(label_values)

    if valid_mask.sum() < 10:
        print(f"跳过: 有效样本数={valid_mask.sum()} < 10")
        return None

    # 提取协变量（age和gender）
    if "age" in scores_df.columns and "gender" in scores_df.columns:
        age_vals = pd.to_numeric(scores_df["age"], errors="coerce").to_numpy()
        gender_vals = pd.to_numeric(scores_df["gender"], errors="coerce").to_numpy()
        age_valid = np.isfinite(age_vals)
        gender_valid = np.isfinite(gender_vals)
        confound_mask = valid_mask & age_valid & gender_valid

        if confound_mask.sum() == valid_mask.sum():
            confounds = np.column_stack((
                age_vals[valid_mask],
                gender_vals[valid_mask]
            ))
        else:
            confounds = None
    else:
        confounds = None

    mats_valid = mats[valid_mask]
    y = label_values[valid_mask].astype(float)

    # 准备特征
    edges_roi_roi_subj = np.transpose(mats_valid, (1, 2, 0))
    num_subj = mats_valid.shape[0]
    num_roi = mats_valid.shape[1]

    train_picked_edges = select_sig_edges(y, edges_roi_roi_subj, num_roi, measurement="pcorr")
    X = train_picked_edges.T

    # 如果使用去交杂，拼接协变量
    if use_deconfound and confounds is not None:
        X = np.hstack((X, confounds))

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    for method_cfg in LINEAR_METHODS:
        method_name = method_cfg["method_name"]
        params = dict(method_cfg["params"])
        if "random_state" in params:
            params["random_state"] = random_state

        base_model = get_regression_model(
            method_cfg["method_type"],
            method_cfg["method_name"],
            **params
        )
        model = clone(base_model)

        # 如果使用去交杂，包装模型
        if use_deconfound and confounds is not None:
            model = DeconfoundWrapper(model, n_confounds=2)

        start = time.time()
        y_pred = cross_val_predict(model, X, y, cv=cv)
        elapsed = time.time() - start

        metrics = evaluate_predictions(y, y_pred)
        metrics.update({
            "method": method_name,
            "use_deconfound": use_deconfound,
            "n_subjects": num_subj,
            "time_sec": elapsed,
        })

        results.append(metrics)
        print(f"  {method_name}: r={metrics['pearson_r']:.4f}, "
              f"MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f} "
              f"({elapsed:.1f}s)")

    return results


def main():
    """主测试函数"""
    dataset_name = "S1200"
    atlas_names = ["bna246", "schaefer200_s1"]
    modal_types = ["SC", "FC"]
    sc_kinds = list(SC_KIND.keys())
    fc_kinds = expand_fc_conn_kinds(list(FC_KIND.keys()))
    all_conn_kinds = {
        "SC": sc_kinds,
        "FC": fc_kinds,
    }
    random_state = 42
    n_splits = 10
    resdir="./res_CVCR"

    # 获取标签列表
    dt_cfg = get_dataset_cfg(dataset_name)
    label_names = dt_cfg["tgt_label_list"][3:]  # 前3个是subject_id, age, gender

    # 选择一个标签进行快速测试（可修改为测试多个）
    test_labels = [label_names[0]] if label_names else []

    if not test_labels:
        print("错误: 未找到可用的标签")
        return

    print(f"\n{'#'*60}")
    print(f"去交杂功能测试")
    print(f"数据集: {dataset_name}")
    print(f"Atlas: {atlas_names}")
    print(f"Modalities: {modal_types}")
    print(f"标签: {test_labels}")
    print(f"#'*60")

    all_results = []

    for atlas_name in atlas_names:
        for modal_type in modal_types:
            conn_kinds = all_conn_kinds.get(modal_type, [])
            for conn_kind in conn_kinds:
                for label_name in test_labels:
                    # 先跑无去交杂版本
                    results_no_dec = run_single_experiment(
                        dataset_name, atlas_name, modal_type, conn_kind,
                        label_name, use_deconfound=False,
                        random_state=random_state, n_splits=n_splits
                    )

                    # 再跑去交杂版本
                    results_dec = run_single_experiment(
                        dataset_name, atlas_name, modal_type, conn_kind,
                        label_name, use_deconfound=True,
                        random_state=random_state, n_splits=n_splits
                    )

                    # 对比结果
                    if results_no_dec and results_dec:
                        print(f"\n--- 对比: {atlas_name}/{modal_type}/{conn_kind}/{label_name} ---")
                        for r_no, r_dec in zip(results_no_dec, results_dec):
                            r_diff = r_dec['pearson_r'] - r_no['pearson_r']
                            print(f"  {r_no['method']}: "
                                  f"r变化={r_diff:+.4f} "
                                  f"({r_no['pearson_r']:.4f} -> {r_dec['pearson_r']:.4f})")

                        # 合并结果
                        for r in results_no_dec:
                            r['experiment'] = f"{atlas_name}/{modal_type}/{conn_kind}/{label_name}"
                            all_results.append(r)
                        for r in results_dec:
                            r['experiment'] = f"{atlas_name}/{modal_type}/{conn_kind}/{label_name}"
                            all_results.append(r)

    # 保存结果
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_path = "./test_deconfound_results.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")

        # 打印汇总
        print("\n" + "="*60)
        print("汇总统计")
        print("="*60)
        summary = df_results.groupby(['method', 'use_deconfound']).agg({
            'pearson_r': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'r2': ['mean', 'std'],
        }).round(4)
        print(summary)


if __name__ == "__main__":
    main()
