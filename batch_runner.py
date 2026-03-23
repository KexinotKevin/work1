import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

import edge_selection
from datasets_base import FC_KIND, SC_KIND, SFC_KIND, load_data, normalize_modal_type
from datasets_cfg import atlases, get_dataset_cfg
from edge_selection import haufe_transform, select_sig_edges
from models import get_regression_model


METHODS = [
    {"method_type": "linear", "method_name": "lasso", "params": {"alpha": 0.05, "max_iter": 5000}},
    {"method_type": "linear", "method_name": "ridge", "params": {"alpha": 1.0}},
    {"method_type": "linear", "method_name": "linear", "params": {}},
    {"method_type": "linear", "method_name": "huber", "params": {"epsilon": 1.35}},
    {
        "method_type": "nonlinear",
        "method_name": "kernel_ridge_rbf",
        "params": {"alpha": 1.0, "gamma": 0.05},
    },
    {
        "method_type": "nonlinear",
        "method_name": "decision_tree",
        "params": {"max_depth": 6, "random_state": 42},
    },
    {"method_type": "nonlinear", "method_name": "gradient_boosting", "params": {"random_state": 42}},
    {
        "method_type": "mlp",
        "method_name": "single_layer_mlp",
        "params": {"hidden_layer_sizes": (64,), "max_iter": 500, "random_state": 42},
    },
]

MODALITY_KIND_MAP = {
    "SC": SC_KIND,
    "FC": FC_KIND,
    "SFCouple": SFC_KIND,
}


def sanitize_name(value):
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    text = "".join(keep).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "unnamed"


def normalize_atlas_name(atlas_name):
    atlas_name_lower = str(atlas_name).lower()
    for valid_name in atlases["ATLAS_NAME"]:
        if valid_name.lower() == atlas_name_lower:
            return valid_name
    raise ValueError(
        f"Unknown atlas '{atlas_name}'. Valid options: {atlases['ATLAS_NAME']}"
    )


def build_combo_stem(dataset_name, atlas_name, modal_type, sub_kind):
    return (
        f"dataset_{sanitize_name(dataset_name)}__"
        f"atlas_{sanitize_name(atlas_name)}__"
        f"modality_{sanitize_name(modal_type)}__"
        f"type_{sanitize_name(sub_kind)}"
    )


def get_label_dirs(base_out_dir, label_name):
    label_dir_name = sanitize_name(label_name)
    stats_dir = os.path.join(base_out_dir, "stats", label_dir_name)
    pics_dir = os.path.join(base_out_dir, "pics", label_dir_name)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(pics_dir, exist_ok=True)
    return stats_dir, pics_dir


def log_progress(dataset_name, modal_type, sub_kind, label_name, stage):
    print(
        f"[dataset={dataset_name}] [modality={modal_type}] "
        f"[type={sub_kind}] [label={label_name}] {stage}"
    )


def get_modality_kind_map(modal_type):
    modal_type = normalize_modal_type(modal_type)
    if modal_type not in MODALITY_KIND_MAP:
        raise ValueError(
            f"Unsupported modality '{modal_type}'. Valid options: {sorted(MODALITY_KIND_MAP)}"
        )
    return MODALITY_KIND_MAP[modal_type]


def load_existing_combo(stats_dir, combo_stem):
    pred_csv = os.path.join(stats_dir, f"pred__{combo_stem}.csv")
    summary_csv = os.path.join(stats_dir, f"summary__{combo_stem}.csv")
    if os.path.exists(pred_csv) and os.path.exists(summary_csv):
        return pd.read_csv(pred_csv), pd.read_csv(summary_csv)
    return None, None


def evaluate_predictions(y_true, y_pred):
    if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        pred_r, pred_p = pearsonr(y_true, y_pred)
    else:
        pred_r, pred_p = np.nan, np.nan
    return {
        "pred_effiency_r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pred_r) if not np.isnan(pred_r) else np.nan,
        "pearson_p": float(pred_p) if not np.isnan(pred_p) else np.nan,
    }


def run_one_dataset(
    base_out_dir,
    dataset_name,
    modal_type,
    atlas_name,
    sub_kind,
    mats_subj_roi_roi,
    labels,
    pred_label_type,
    random_state=42,
    n_splits=10,
):
    atlas_name = normalize_atlas_name(atlas_name)
    modal_type = normalize_modal_type(modal_type)
    combo_stem = build_combo_stem(dataset_name, atlas_name, modal_type, sub_kind)
    stats_dir, _pics_dir = get_label_dirs(base_out_dir, pred_label_type)

    existing_pred_df, existing_summary_df = load_existing_combo(stats_dir, combo_stem)
    if existing_pred_df is not None and existing_summary_df is not None:
        log_progress(dataset_name, modal_type, sub_kind, pred_label_type, "skip existing csv")
        return existing_pred_df, existing_summary_df

    log_progress(dataset_name, modal_type, sub_kind, pred_label_type, "prepare features")
    edges_roi_roi_subj = np.transpose(mats_subj_roi_roi, (1, 2, 0))
    num_subj = mats_subj_roi_roi.shape[0]
    num_roi = mats_subj_roi_roi.shape[1]

    if num_subj < n_splits:
        raise ValueError(
            f"Number of subjects ({num_subj}) is smaller than n_splits ({n_splits})."
        )

    edge_selection.pcorr_res = np.zeros((num_roi, num_roi, 2), dtype=float)
    train_picked_edges = select_sig_edges(labels, edges_roi_roi_subj, num_roi, measurement="pcorr")
    X = train_picked_edges.T
    y = labels.astype(float)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    per_subject_rows = []
    summary_rows = []

    for method_cfg in METHODS:
        method_name = method_cfg["method_name"]
        log_progress(dataset_name, modal_type, sub_kind, pred_label_type, f"cv fit start: {method_name}")
        params = dict(method_cfg["params"])
        if "random_state" in params:
            params["random_state"] = random_state

        base_model = get_regression_model(method_cfg["method_type"], method_cfg["method_name"], **params)
        model = clone(base_model)
        y_pred = cross_val_predict(model, X, y, cv=cv)
        metrics = evaluate_predictions(y, y_pred)

        log_progress(dataset_name, modal_type, sub_kind, pred_label_type, f"full fit start: {method_name}")
        fitted_model = clone(base_model)
        fitted_model.fit(X, y)

        haufe_csv = ""
        haufe_status = "not_supported"
        haufe_error = ""

        if hasattr(fitted_model, "coef_"):
            try:
                model_coef = np.asarray(fitted_model.coef_).reshape(-1)
                haufe_mat = haufe_transform(train_picked_edges, y, model_coef, num_nodes=num_roi)
                haufe_csv = os.path.join(
                    stats_dir,
                    f"haufe__{combo_stem}__method_{sanitize_name(method_name)}.csv",
                )
                pd.DataFrame(haufe_mat).to_csv(haufe_csv, index=False)
                haufe_status = "ok"
                log_progress(dataset_name, modal_type, sub_kind, pred_label_type, f"haufe saved: {method_name}")
            except Exception as exc:
                haufe_status = "failed"
                haufe_error = str(exc)
                log_progress(dataset_name, modal_type, sub_kind, pred_label_type, f"haufe failed: {method_name}")

        for subj_i in range(num_subj):
            per_subject_rows.append(
                {
                    "dataset": dataset_name,
                    "modality": modal_type,
                    "atlas": atlas_name,
                    "conn_kind": sub_kind,
                    "subject_index": int(subj_i),
                    "method_type": method_cfg["method_type"],
                    "method_name": method_name,
                    "pred_label_type": pred_label_type,
                    "true_label": float(y[subj_i]),
                    "pred_scores": float(y_pred[subj_i]),
                    "pred_effiency_r2": metrics["pred_effiency_r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "pearson_r": metrics["pearson_r"],
                    "pearson_p": metrics["pearson_p"],
                }
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "modality": modal_type,
                "atlas": atlas_name,
                "conn_kind": sub_kind,
                "method_type": method_cfg["method_type"],
                "method_name": method_name,
                "pred_label_type": pred_label_type,
                "n_subjects": int(num_subj),
                "cv_folds": int(n_splits),
                "haufe_status": haufe_status,
                "haufe_csv": haufe_csv,
                "haufe_error": haufe_error,
                **metrics,
            }
        )
        log_progress(dataset_name, modal_type, sub_kind, pred_label_type, f"done: {method_name}")

    pred_df = pd.DataFrame(per_subject_rows)
    summary_df = pd.DataFrame(summary_rows)
    pred_csv = os.path.join(stats_dir, f"pred__{combo_stem}.csv")
    summary_csv = os.path.join(stats_dir, f"summary__{combo_stem}.csv")

    pred_df.to_csv(pred_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    log_progress(dataset_name, modal_type, sub_kind, pred_label_type, "csv saved")
    return pred_df, summary_df


def run_modality_batch(
    dataset_name="HCD",
    atlas_name=None,
    modal_type="SC",
    sub_kind=None,
    modality_name=None,
    atlas_names=None,
    conn_kinds=None,
    label_names=None,
    random_state=42,
):
    base_out_dir = f"./res_{dataset_name}"
    os.makedirs(os.path.join(base_out_dir, "stats"), exist_ok=True)
    os.makedirs(os.path.join(base_out_dir, "pics"), exist_ok=True)
    dt_cfg = get_dataset_cfg(dataset_name)
    modal_type = normalize_modal_type(modality_name if modality_name is not None else modal_type)
    if atlas_names is None and atlas_name is not None:
        atlas_names = [atlas_name]
    atlas_names = atlas_names or ["aal116", "bna246", "schaefer200_S1"]
    atlas_names = [normalize_atlas_name(name) for name in atlas_names]
    kind_map = get_modality_kind_map(modal_type)
    if conn_kinds is None and sub_kind is not None:
        conn_kinds = [sub_kind]
    conn_kinds = conn_kinds or list(kind_map.keys())
    label_names = label_names or dt_cfg["tgt_label_list"][3:]

    all_pred = []
    all_summary = []

    for atlas_name in atlas_names:
        for conn_kind in conn_kinds:
            log_progress(dataset_name, modal_type, conn_kind, "ALL", f"load data: atlas={atlas_name}")
            dataset = load_data(
                dataset_name=dataset_name,
                dt_cfg=dt_cfg,
                atlas_name=atlas_name,
                modal_type=modal_type,
                sub_kind=conn_kind,
            )
            mats = dataset["nets"][atlas_name][conn_kind]
            scores_df = dataset["scores"].copy()

            for label_name in label_names:
                label_values = pd.to_numeric(scores_df[label_name], errors="coerce").to_numpy()
                valid_mask = np.isfinite(label_values)
                if valid_mask.sum() < 10:
                    log_progress(dataset_name, modal_type, conn_kind, label_name, "skip: fewer than 10 valid subjects")
                    continue

                pred_df, summary_df = run_one_dataset(
                    base_out_dir=base_out_dir,
                    dataset_name=dataset_name,
                    modal_type=modal_type,
                    atlas_name=atlas_name,
                    sub_kind=conn_kind,
                    mats_subj_roi_roi=mats[valid_mask],
                    labels=label_values[valid_mask],
                    pred_label_type=label_name,
                    random_state=random_state,
                    n_splits=10,
                )
                all_pred.append(pred_df)
                all_summary.append(summary_df)

    pred_all_df = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    summary_all_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    pred_all_df.to_csv(
        os.path.join(base_out_dir, "stats", f"pred__{dataset_name}__{sanitize_name(modal_type)}_all_combinations.csv"),
        index=False,
    )
    summary_all_df.to_csv(
        os.path.join(
            base_out_dir,
            "stats",
            f"summary__{dataset_name}__{sanitize_name(modal_type)}_all_combinations.csv",
        ),
        index=False,
    )
    log_progress(dataset_name, modal_type, "ALL", "ALL", "batch completed")
    return pred_all_df, summary_all_df


def run_hcd_batch(dataset_name="HCD", atlas_names=None, sc_types=None, label_names=None, random_state=42):
    return run_modality_batch(
        dataset_name=dataset_name,
        modal_type="SC",
        atlas_names=atlas_names,
        conn_kinds=sc_types,
        label_names=label_names,
        random_state=random_state,
    )
