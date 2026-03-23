import os
import numpy as np
import pandas as pd

from datasets_cfg import get_dataset_cfg

std_names = ['subject_id','gender','age']

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
    scores = scores[std_names+dt_cfg["tgt_label_list"][3:]]
    scores = scores.sort_values(by="subject_id")
    scores["subject_id"] = scores["subject_id"].astype(str)
    subjids = scores["subject_id"].astype(str).tolist()
    subjids.sort()
    
    # 快速验证
    assert len(scores) == len(subjids), "长度不一致"
    assert set(scores['subject_id'].astype(str)) == set(subjids), "内容不一致"
    assert scores['subject_id'].astype(str).tolist() == subjids, "顺序不一致"
    print("✓ 验证通过")

    nets, valid_subjids = load_conn(
        subjids=subjids,
        conn_dir=conn_dir,
        atlas_name=atlas_names,
        conn_type=modal_type,
        conn_kind=sub_kinds,
    )

    scores = scores[scores["subject_id"].isin(valid_subjids)].copy()
    scores["subject_id"] = scores["subject_id"].astype(str)
    # scores = scores.set_index("subject_id").loc[valid_subjids].reset_index()

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

    nets = {}
    for atlas, kind_to_subj_file in atlas_to_kind_subj_file.items():
        nets[atlas] = {}
        for k in conn_kind:
            resolved_kind = resolve_conn_filename(conn_type, k)
            mats = []
            subj_to_file = kind_to_subj_file[k]
            for sid in valid_subjids:
                mat = pd.read_csv(subj_to_file[sid], header=None).values
                mats.append(mat)
            mats_stack = np.stack(mats, axis=0)
            nets[atlas][k] = mats_stack
            nets[atlas][resolved_kind] = mats_stack

    return nets, valid_subjids

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
    df_renamed['gender'] = df_renamed['gender'].replace(['F', 2], 'Female')
    df_renamed['gender'] = df_renamed['gender'].replace(['M', 1], 'Male')
    
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
