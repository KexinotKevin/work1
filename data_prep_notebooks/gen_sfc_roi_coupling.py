import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
WORK1_DIR = os.path.dirname(CUR_DIR)
if WORK1_DIR not in sys.path:
    sys.path.append(WORK1_DIR)

from datasets_base import load_conn, load_scores, std_names
from datasets_cfg import get_dataset_cfg


def _safe_offdiag_zscore(M, eps=1e-12):
    n = M.shape[0]
    out = M.copy().astype(float)
    mask = ~np.eye(n, dtype=bool)
    x = out[mask]
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        sd = 1.0
    out[mask] = (x - mu) / sd
    np.fill_diagonal(out, 0.0)
    return out


def _sym_matrix_exp(M):
    # Symmetric matrix exponential via eigendecomposition
    w, v = np.linalg.eigh((M + M.T) / 2.0)
    ew = np.exp(w)
    return (v * ew) @ v.T


def simulate_sc_matrix(n_roi, rng, sparsity=0.75):
    # Random weighted undirected SC with a weak ring backbone for connectivity
    W = rng.uniform(0.0, 1.0, size=(n_roi, n_roi))
    W = (W + W.T) / 2.0

    keep = rng.uniform(0.0, 1.0, size=(n_roi, n_roi)) > sparsity
    keep = np.triu(keep, 1)
    keep = keep + keep.T

    A = W * keep

    # ring backbone
    for i in range(n_roi):
        j = (i + 1) % n_roi
        A[i, j] = max(A[i, j], 0.2)
        A[j, i] = A[i, j]

    np.fill_diagonal(A, 0.0)
    return A


def communicability(A, eps=1e-12):
    s = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(s, eps))
    A_norm = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    G = _sym_matrix_exp(A_norm)
    np.fill_diagonal(G, 0.0)
    return G


def flow_graph(A, t=1.0, eps=1e-12):
    # Symmetric normalized Laplacian implementation
    s = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(s, eps))
    A_norm = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    L_sym = np.eye(A.shape[0]) - A_norm
    E = _sym_matrix_exp(-t * L_sym)
    G = E * s[None, :]
    G = (G + G.T) / 2.0
    np.fill_diagonal(G, 0.0)
    return G


def mfpt(A, eps=1e-12):
    # MFPT for random walk with transition matrix P = D^{-1}A
    n = A.shape[0]
    s = A.sum(axis=1)
    P = A / np.maximum(s[:, None], eps)

    T = np.zeros((n, n), dtype=float)
    I = np.eye(n - 1)

    for target in range(n):
        idx = [k for k in range(n) if k != target]
        Q = P[np.ix_(idx, idx)]
        # (I - Q) h = 1
        h = np.linalg.solve(I - Q, np.ones(n - 1))
        T[idx, target] = h

    np.fill_diagonal(T, 0.0)
    return T


def build_fc_from_sc(A, rng, noise_scale=0.25):
    # Build synthetic FC from SC communication predictors + noise
    C = communicability(A)
    F = flow_graph(A, t=1.0)
    M = mfpt(A)

    A_z = _safe_offdiag_zscore(A)
    C_z = _safe_offdiag_zscore(C)
    F_z = _safe_offdiag_zscore(F)
    M_inv_z = _safe_offdiag_zscore(1.0 / (1.0 + M))

    FC = 0.45 * A_z + 0.25 * C_z + 0.20 * F_z + 0.10 * M_inv_z
    FC += noise_scale * rng.normal(size=FC.shape)
    FC = (FC + FC.T) / 2.0
    np.fill_diagonal(FC, 0.0)
    return FC


def adjusted_r2(y, y_hat, n_pred):
    n = y.size
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    denom = n - n_pred - 1
    if denom <= 0:
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / denom


def sc_fc_coupling_one_subject(A, FC):
    C = communicability(A)
    F = flow_graph(A, t=1.0)
    M = mfpt(A)
    M_inv = 1.0 / (1.0 + M)

    n = A.shape[0]
    node_adj_r2 = np.zeros(n, dtype=float)

    for i in range(n):
        mask = np.arange(n) != i
        y = FC[i, mask]
        X = np.column_stack([
            A[i, mask],
            C[i, mask],
            F[i, mask],
            M_inv[i, mask],
        ])

        # z-score each predictor column
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat = X_aug @ beta
        node_adj_r2[i] = adjusted_r2(y, y_hat, n_pred=X.shape[1])

    return node_adj_r2


def _prepare_subject_ids(dataset_name, dt_cfg):
    scores_df = load_scores(dataset_name, dt_cfg)
    score_cols = std_names + dt_cfg["tgt_label_list"][3:]
    scores_df = scores_df[score_cols].copy()
    scores_df = scores_df.sort_values(by="subject_id")
    scores_df["subject_id"] = scores_df["subject_id"].astype(str)
    subjids = scores_df["subject_id"].tolist()
    subjids.sort()
    return subjids


def _align_sc_fc_mats(subjids, conn_dir, atlas_name, sc_kind, fc_kind):
    sc_nets, sc_valid_subjids = load_conn(
        subjids=subjids,
        conn_dir=conn_dir,
        atlas_name=[atlas_name],
        conn_type="SC",
        conn_kind=[sc_kind],
    )
    fc_nets, fc_valid_subjids = load_conn(
        subjids=subjids,
        conn_dir=conn_dir,
        atlas_name=[atlas_name],
        conn_type="FC",
        conn_kind=[fc_kind],
    )

    common_subjids = [sid for sid in subjids if sid in set(sc_valid_subjids) and sid in set(fc_valid_subjids)]
    if not common_subjids:
        raise ValueError("No subject has both SC and FC files for the selected atlas and kinds.")

    sc_idx_map = {sid: idx for idx, sid in enumerate(sc_valid_subjids)}
    fc_idx_map = {sid: idx for idx, sid in enumerate(fc_valid_subjids)}
    sc_indices = [sc_idx_map[sid] for sid in common_subjids]
    fc_indices = [fc_idx_map[sid] for sid in common_subjids]

    sc_mats = sc_nets[atlas_name][sc_kind][sc_indices]
    fc_mats = fc_nets[atlas_name][fc_kind][fc_indices]

    if sc_mats.shape != fc_mats.shape:
        raise ValueError(f"SC and FC shape mismatch: SC={sc_mats.shape}, FC={fc_mats.shape}")
    if sc_mats.shape[1] != sc_mats.shape[2]:
        raise ValueError(f"Expected square matrices, got shape {sc_mats.shape}")
    return common_subjids, sc_mats, fc_mats


def compute_sfc_roi_coupling_dataframe(
    dataset_name,
    atlas_name,
    sc_kind,
    fc_kind="pcc_rest",
    conn_dir=None,
    scores_path=None,
):
    dt_cfg = get_dataset_cfg(dataset_name)
    if conn_dir:
        dt_cfg["conn_dir"] = conn_dir
    if scores_path:
        dt_cfg["scores_path"] = scores_path

    subjids = _prepare_subject_ids(dataset_name, dt_cfg)
    common_subjids, sc_mats, fc_mats = _align_sc_fc_mats(
        subjids=subjids,
        conn_dir=dt_cfg["conn_dir"],
        atlas_name=atlas_name,
        sc_kind=sc_kind,
        fc_kind=fc_kind,
    )

    num_subj, num_roi, _ = sc_mats.shape
    rows = np.zeros((num_subj, num_roi), dtype=float)
    for i in range(num_subj):
        rows[i] = sc_fc_coupling_one_subject(sc_mats[i], fc_mats[i])

    roi_cols = [f"roi_idx_{i}" for i in range(num_roi)]
    out_df = pd.DataFrame(rows, columns=roi_cols)
    out_df.insert(0, "subject_id", common_subjids)
    return out_df


def build_default_output_csv(dataset_name, atlas_name, sc_kind, fc_kind):
    fname = (
        f"sfc_roi_coupling__dataset_{dataset_name}"
        f"__atlas_{atlas_name}__sc_{sc_kind}__fc_{fc_kind}.csv"
    )
    return os.path.join(CUR_DIR, fname)


def main():
    parser = argparse.ArgumentParser(description="Compute ROI-level SC-FC coupling and save to CSV.")
    parser.add_argument("--dataset", type=str, default="HCD", help="Dataset key in datasets_cfg.py, e.g. HCD/ABCD/S1200")
    parser.add_argument("--atlas", type=str, default="aal116", help="Atlas name")
    parser.add_argument("--sc-kind", type=str, default="fiber_count", help="SC kind key in datasets_base.SC_KIND")
    parser.add_argument("--fc-kind", type=str, default="pcc_rest", help="FC kind key in datasets_base.FC_KIND")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--conn-dir", type=str, default=None, help="Override conn_dir in dataset config")
    parser.add_argument("--scores-path", type=str, default=None, help="Override scores_path in dataset config")
    args = parser.parse_args()

    out_df = compute_sfc_roi_coupling_dataframe(
        dataset_name=args.dataset,
        atlas_name=args.atlas,
        sc_kind=args.sc_kind,
        fc_kind=args.fc_kind,
        conn_dir=args.conn_dir,
        scores_path=args.scores_path,
    )
    out_csv = args.out_csv or build_default_output_csv(args.dataset, args.atlas, args.sc_kind, args.fc_kind)
    out_df.to_csv(out_csv, index=False)
    print(f"[ok] saved csv: {out_csv}")
    print(f"[ok] subjects: {len(out_df)}, rois: {out_df.shape[1] - 1}")


if __name__ == "__main__":
    main()
