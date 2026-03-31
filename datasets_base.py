import os
from collections import Counter

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from datasets_cfg import get_dataset_cfg

std_names = ['subject_id','gender','age']


def normalize_subject_id(sid, dataset_name=None):
    """
    标准化 subject ID：将各种可能的 NDAR ID 格式统一为 CSV 中的原始格式。
    ABCD 的 scores CSV 使用 "NDAR_INV..." 格式，而 network 目录使用 "NDARINV..." 格式。
    """
    s = str(sid)
    if dataset_name == "ABCD":
        s = s.replace("NDARINV", "NDAR_INV")
    return s


def _normalize_subject_ids(subjids, dataset_name=None):
    """将 subject IDs 标准化为 CSV 中的原始格式（用于 scores 匹配）。"""
    return [normalize_subject_id(s, dataset_name) for s in subjids]


def _denormalize_subject_ids(subjids, dataset_name=None):
    """将 subject IDs 反标准化为 network 目录中的格式（用于文件路径查找）。"""
    result = []
    for sid in subjids:
        s = str(sid)
        if dataset_name == "ABCD":
            s = s.replace("NDAR_INV", "NDARINV")
        result.append(s)
    return result


"""
dataset_cfg[dataset_type] = {
        "conn_dir": "",
        "scores_path": "",
        "demographics_path":"",
        "tgt_label_list": [],
        "tgt_demo_list": []
    }
"""
SC_KIND = {
    "fiber_length": "connectome_mean_length_10M.csv",
    "FA": "connectome_mean_FA_10M.csv",
    "fiber_bundle_capacity": "connectome_sift2_fbc_10M.csv",
    "fiber_count": "connectome_streamline_count_10M.csv",
}

FC_KIND = {
    "pcc_rest": "pFC.csv",
}

FC_CONN_MASK = {
    "pcc_rest_pos": "positive connections only",
    "pcc_rest_neg": "negative connections only",
}


def _apply_fc_conn_mask(mats_stack, sub_kind):
    """对 FC 矩阵应用掩码：仅保留正连接或仅保留负连接。"""
    if not isinstance(mats_stack, np.ndarray) or mats_stack.ndim != 3:
        raise ValueError(f"Expected 3D array for FC matrices, got shape {mats_stack.shape if isinstance(mats_stack, np.ndarray) else type(mats_stack)}")

    if sub_kind.endswith("_pos"):
        masked = np.where(mats_stack > 0, mats_stack, 0)
        print(f"  [mask] Applied positive-only mask: {mats_stack.shape}")
    elif sub_kind.endswith("_neg"):
        masked = np.where(mats_stack < 0, mats_stack, 0)
        print(f"  [mask] Applied negative-only mask: {mats_stack.shape}")
    else:
        masked = mats_stack
    return masked

SFC_KIND={
    "sc_roi_fitted": "",
}

VALID_TYPE_TO_KIND = {
    "SC": SC_KIND,
    "FC": FC_KIND,
    "SFCouple": SFC_KIND,
}


def normalize_modal_type(modal_type):
    modal_type_text = str(modal_type).strip()
    modal_type_upper = modal_type_text.upper()
    if modal_type_upper == "SC":
        return "SC"
    if modal_type_upper == "FC":
        return "FC"
    if modal_type_text == "SFCouple":
        return "SFCouple"
    raise ValueError("modal_type must be one of ['SC', 'FC', 'SFCouple'].")


def ensure_list(value, field_name):
    if value is None:
        raise ValueError(f"{field_name} is required.")
    if isinstance(value, (list, tuple, set)):
        values = [item for item in value if item is not None]
    else:
        values = [value]
    if not values:
        raise ValueError(f"{field_name} must be non-empty.")
    return values


def resolve_conn_filename(modal_type, sub_kind):
    modal_type = normalize_modal_type(modal_type)
    sub_kind = str(sub_kind).strip()
    if not sub_kind:
        raise ValueError("sub_kind must be non-empty.")

    if modal_type == "FC":
        if sub_kind in FC_CONN_MASK:
            return resolve_conn_filename("FC", sub_kind.rsplit("_pos", 1)[0].rsplit("_neg", 1)[0])

    kind_map = VALID_TYPE_TO_KIND[modal_type]
    mapped_name = kind_map.get(sub_kind, "")
    if mapped_name:
        return mapped_name

    if modal_type == "FC":
        if sub_kind == "pFC.csv":
            return sub_kind
        raise ValueError("For FC, sub_kind must be 'pcc_rest' or file name 'pFC.csv'.")

    return sub_kind


def load_data(
    conn_dir=None,
    atlas_name=None,
    conn_type=None,
    conn_kind=None,
    dt_name=None,
    dt_cfg=None,
    *,
    dataset_name=None,
    atlas_names=None,
    modal_type=None,
    sub_kind=None,
):
    dataset_name = dataset_name or dt_name
    if not dataset_name:
        raise ValueError("dataset_name or dt_name is required.")

    dt_cfg = dt_cfg or get_dataset_cfg(dataset_name)
    conn_dir = conn_dir or dt_cfg.get("conn_dir", "")
    atlas_names = atlas_name if atlas_name is not None else atlas_names
    sub_kinds = conn_kind if conn_kind is not None else sub_kind
    modal_type = conn_type if conn_type is not None else modal_type

    atlas_names = ensure_list(atlas_names, "atlas_name")
    sub_kinds = ensure_list(sub_kinds, "sub_kind")
    modal_type = normalize_modal_type(modal_type)

    scores = load_scores(dataset_name, dt_cfg)
    scores = scores[std_names + dt_cfg["tgt_label_list"][3:]]
    scores = scores.sort_values(by="subject_id")
    scores["subject_id"] = scores["subject_id"].astype(str)
    subjids = scores["subject_id"].astype(str).tolist()
    subjids.sort()

    assert len(scores) == len(subjids), "长度不一致"
    assert set(scores['subject_id'].astype(str)) == set(subjids), "内容不一致"
    assert scores['subject_id'].astype(str).tolist() == subjids, "顺序不一致"
    print("✓ 验证通过")

    # Denormalize for directory matching (ABCD uses NDARINV without underscore)
    net_subjids = _denormalize_subject_ids(subjids, dataset_name)

    nets, valid_net_subjids = load_conn(
        subjids=net_subjids,
        conn_dir=conn_dir,
        atlas_name=atlas_names,
        conn_type=modal_type,
        conn_kind=sub_kinds,
    )

    # Re-normalize for scores matching, then filter scores
    valid_subjids = _normalize_subject_ids(valid_net_subjids, dataset_name)
    scores = scores[scores["subject_id"].isin(valid_subjids)].copy()
    scores["subject_id"] = scores["subject_id"].astype(str)

    return {"scores": scores, "nets": nets}

def load_conn(
    subjids,
    conn_dir=None,
    atlas_name=None,
    conn_type=None,
    conn_kind=None,
    *,
    modal_type=None,
    sub_kind=None,
):
    atlas_name = ensure_list(atlas_name, "atlas_name")
    conn_kind = ensure_list(
        conn_kind if conn_kind is not None else sub_kind,
        "sub_kind",
    )
    conn_type = normalize_modal_type(conn_type if conn_type is not None else modal_type)

    if not conn_dir:
        raise ValueError("conn_dir is empty.")
    subjids.sort()
    subjids = [str(s) for s in subjids]
    atlas_to_kind_subj_file = {}
    common_subjids = set(subjids)

    for atlas in atlas_name:
        kind_to_subj_file = {}
        atlas_subjid_set = set(subjids)

        for k in conn_kind:
            kind = resolve_conn_filename(conn_type, k)
            subj_to_file = {}
            for sid in subjids:
                fpath = os.path.join(conn_dir, atlas, sid, conn_type, kind)
                if not os.path.isfile(fpath):
                    fpath_csv = fpath + ".csv"
                    if os.path.isfile(fpath_csv):
                        fpath = fpath_csv
                    else:
                        continue
                subj_to_file[sid] = fpath

            kind_to_subj_file[k] = subj_to_file
            atlas_subjid_set &= set(subj_to_file.keys())

        atlas_to_kind_subj_file[atlas] = kind_to_subj_file
        common_subjids &= atlas_subjid_set

    valid_subjids = [sid for sid in subjids if sid in common_subjids]
    if not valid_subjids:
        raise ValueError("No subject has all requested atlas/kind files.")

    current_valid = list(valid_subjids)
    nets = {}

    def _subset_all_nets(old_order, new_order):
        if len(new_order) == len(old_order) and new_order == old_order:
            return
        keep_idx = [old_order.index(s) for s in new_order]
        for atlas in list(nets.keys()):
            replaced = {}
            for kk in list(nets[atlas].keys()):
                arr = nets[atlas][kk]
                if not isinstance(arr, np.ndarray) or arr.ndim != 3:
                    continue
                oid = id(arr)
                if oid in replaced:
                    nets[atlas][kk] = replaced[oid]
                else:
                    new_arr = arr[keep_idx]
                    replaced[oid] = new_arr
                    nets[atlas][kk] = new_arr

    for atlas, kind_to_subj_file in atlas_to_kind_subj_file.items():
        nets[atlas] = {}
        for k in conn_kind:
            resolved_kind = resolve_conn_filename(conn_type, k)
            subj_to_file = kind_to_subj_file[k]

            def _read_csv(sid):
                return pd.read_csv(subj_to_file[sid], header=None).values

            empty_sids = set(sid for sid in subj_to_file if os.path.getsize(subj_to_file[sid]) == 0)
            if empty_sids:
                print(f"  [warn] Skipping {len(empty_sids)} subject(s) with empty files for atlas={atlas} kind={k}")
                for sid in empty_sids:
                    del subj_to_file[sid]

            current_valid = [sid for sid in current_valid if sid not in empty_sids]
            if not current_valid:
                raise ValueError(
                    f"No connectivity matrices left after filtering empty files for atlas={atlas} kind={k}."
                )

            with ThreadPoolExecutor(max_workers=min(32, len(current_valid))) as pool:
                mats = list(pool.map(_read_csv, current_valid))

            shape_counts = Counter(tuple(m.shape) for m in mats)
            if len(shape_counts) > 1:
                mode_shape = shape_counts.most_common(1)[0][0]
                kept = [(sid, m) for sid, m in zip(current_valid, mats) if tuple(m.shape) == mode_shape]
                old_order = current_valid
                current_valid = [p[0] for p in kept]
                mats = [p[1] for p in kept]
                _subset_all_nets(old_order, current_valid)
            if not mats:
                raise ValueError(
                    f"No connectivity matrices left after shape filter for atlas={atlas} kind={k}. "
                    f"Shapes seen: {dict(shape_counts)}"
                )

            mats_stack = np.stack(mats, axis=0)
            if conn_type == "FC" and k in FC_CONN_MASK:
                mats_stack = _apply_fc_conn_mask(mats_stack, k)
            nets[atlas][k] = mats_stack
            nets[atlas][resolved_kind] = mats_stack

    if not current_valid:
        raise ValueError("No subject left after connectivity matrix shape alignment.")

    return nets, current_valid

def load_scores(dt_name, dt_cfg):
    scores_path = dt_cfg.get("scores_path", "")
    if not scores_path:
        raise ValueError(
            f"scores_path of {dt_name} is invalid. please check dataset_cfg file."
        )

    df = pd.read_csv(scores_path)
    col_list = dt_cfg.get("tgt_label_list", [])
    if len(col_list) < 3:
        raise ValueError(f"tgt_label_list of {dt_name} is invalid.")

    df = _standardize_columns(df, dt_cfg["tgt_label_list"])
    tgt_cols = dt_cfg["tgt_label_list"][3:]
    if dt_name == "ABCD":
        df = df[df['gender'] != 3]
    # df_long = load_data_and_to_long(df, tgt_cols)

    return df

def _standardize_columns(df, tgtlabels):
    """
    将 tgtlabels 对应的列重命名为标准名称
    
    Parameters:
        df: 输入 DataFrame
        tgtlabels: 包含前三个元素依次对应 subject_id, age, sex 的列表
    
    Returns:
        重命名后的 DataFrame
    """
    # 创建映射：原列名 → 标准列名
    rename_map = {}
    for i, std_name in enumerate(std_names):
        if i < len(tgtlabels) and tgtlabels[i] in df.columns:
            rename_map[tgtlabels[i]] = std_name
    
    print(f"重命名映射: {rename_map}")
    
    # 重命名
    df_renamed = df.rename(columns=rename_map)
    
    # 将性别转换为 0/1 数值型，并确保 age 是浮点数，以便后续进行线性回归
    df_renamed['gender'] = df_renamed['gender'].replace(['F', 'Female', 2], 0)
    df_renamed['gender'] = df_renamed['gender'].replace(['M', 'Male', 1], 1)
    df_renamed['gender'] = df_renamed['gender'].astype(float)
    
    #【更新】增加对不同数据集年龄格式的兼容性处理
    if 'age' in df_renamed.columns:
        # 定义内部解析函数，兼容三种数据集格式
        def _parse_age_robustly(val):
            if pd.isna(val): return val
            s = str(val).strip()
            if '-' in s: # 针对 S1200: 将 "26-30" 转换为中值 28.0
                try:
                    low, high = s.split('-')
                    return (float(low) + float(high)) / 2
                except: return np.nan
            elif '+' in s: # 针对 S1200: 将 "36+" 转换为下界数值 36.0
                try:
                    return float(s.replace('+', ''))
                except: return np.nan
            else: # 针对 ABCD ("9") 和 HCD (25.5) 的标准数值格式
                return pd.to_numeric(s, errors='coerce')
        
        df_renamed['age'] = df_renamed['age'].apply(_parse_age_robustly).astype(float)
    
    # 检查哪些标准列缺失
    missing = set(std_names) - set(rename_map.values())
    if missing:
        print(f"警告: 缺少列 {missing}")
    
    return df_renamed

def load_data_and_to_long(df,tgt_labels):
    df_long = df.melt(
        id_vars=std_names,  # 保持不变的列
        value_vars=tgt_labels,                            # 需要转换的列
        var_name='intell_category',                       # 新列名：原列名
        value_name='scores'                         # 新列名：原值
    )
    df_long['scores'] = pd.to_numeric(df_long['scores'], errors="coerce")
    df_long = df_long[df_long['scores']<500]
    return df_long
