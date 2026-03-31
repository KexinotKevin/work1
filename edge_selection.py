import math
import os
import json
import time

import numpy as np
from scipy.stats import t as t_dist

# #region agent debug logs (optional; never fail training if log path is unavailable)
def _dlog(hid, loc, msg, data=None):
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "debug-agent.log")
        with open(path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": f"{hid}_{os.getpid()}",
                        "timestamp": time.time() * 1000,
                        "hypothesisId": hid,
                        "location": loc,
                        "message": msg,
                        "data": data,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


# #endregion


def _infer_num_nodes(num_edges):
    _dlog("H5", "edge_selection:infer", "infer_num_nodes",
          {"num_edges": int(num_edges)})
    num_nodes = (1 + math.sqrt(1 + 8 * num_edges)) / 2
    if not float(num_nodes).is_integer():
        raise ValueError(
            f"Cannot infer node count from {num_edges} edges. "
            "Expected a triangular-number edge vector."
        )
    return int(num_nodes)


def haufe_transform(train_picked_edges, lb_col, model_coef, num_nodes=None):
    # train_picked_edges shape: (n_edges, n_subj)
    _dlog("H3", "edge_selection:haufe", "enter haufe_transform",
          {"te_shape": list(train_picked_edges.shape),
           "lb_shape": list(lb_col.shape),
           "lb_nan": int(np.isnan(lb_col).sum()), "lb_zero_std": float(np.std(lb_col) < 1e-12)})
    cov_x = np.cov(train_picked_edges)
    cov_y = np.cov(lb_col)
    _dlog("H3", "edge_selection:haufe_cov", "cov values",
          {"cov_x_shape": list(cov_x.shape), "cov_y": float(cov_y),
           "cov_y_zero": bool(abs(cov_y) < 1e-12)})
    haufe_weights = np.matmul(cov_x, model_coef) * (1 / cov_y)
    if num_nodes is None:
        num_nodes = _infer_num_nodes(haufe_weights.shape[0])
    haufe_all_weights = reshape_feat_to_net(haufe_weights, num_nodes)
    return haufe_all_weights


def rearrange_edges(graphs, num_nodes, num_subjs):
    rows, cols = np.triu_indices(num_nodes, k=1)
    return graphs[rows, cols, :]


def reshape_feat_to_net(feature_vector, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes))
    triu_indices = np.triu_indices(num_nodes, k=1)
    matrix[triu_indices] = feature_vector
    matrix = matrix + matrix.T
    return matrix


def select_sig_edges(lb_list, edges, num_nodes, measurement="pcorr"):
    # edges shape: (num_nodes, num_nodes, num_subj)
    # lb_list shape: (num_subj,)
    _dlog("H1", "edge_selection:1", "enter select_sig_edges",
          {"lb_shape": list(lb_list.shape), "edges_shape": list(edges.shape),
           "num_nodes": int(num_nodes), "n_subj": int(lb_list.shape[0])})
    n_subj = lb_list.shape[0]

    # Flatten edges to (num_edges, num_subj) where num_edges = num_nodes * num_nodes
    edges_flat = edges.reshape(-1, n_subj)
    _dlog("H2", "edge_selection:2", "edges_flat reshape",
          {"edges_flat_shape": list(edges_flat.shape),
           "expected_flat": int(num_nodes * num_nodes)})

    if measurement == "pcorr":
        # Vectorized Pearson correlation: corr(edge_i, lb) for all edges at once
        # r = cov(edge_i, lb) / (std(edge_i) * std(lb))
        lb_mean = lb_list.mean()
        lb_std = lb_list.std(ddof=0)
        lb_centered = lb_list - lb_mean

        edge_mean = edges_flat.mean(axis=1, keepdims=True)
        edge_std = edges_flat.std(axis=1, ddof=0)
        edge_centered = edges_flat - edge_mean

        denom = edge_std * lb_std * n_subj
        denom = np.maximum(denom, 1e-12)
        r_all = np.dot(edge_centered, lb_centered) / denom
        # Compute p-values: under H0, t = r*sqrt(n-2)/sqrt(1-r^2) follows t(df=n-2)
        t_stat = r_all * np.sqrt(n_subj - 2) / np.sqrt(np.maximum(1 - r_all**2, 1e-12))
        p_all = 2 * t_dist.sf(np.abs(t_stat), n_subj - 2)
        r_matrix = r_all.reshape(num_nodes, num_nodes)
        p_matrix = p_all.reshape(num_nodes, num_nodes)

    elif measurement == "rankcorr":
        # rankdata along axis=1 (subjects), then compute Pearson correlation
        from scipy.stats import rankdata
        lb_ranked = rankdata(lb_list)
        edges_ranked = rankdata(edges_flat, axis=1)
        lb_mean = lb_ranked.mean()
        lb_std = lb_ranked.std(ddof=0)
        lb_centered = lb_ranked - lb_mean
        edge_mean = edges_ranked.mean(axis=1, keepdims=True)
        edge_std = edges_ranked.std(axis=1, ddof=0)
        edge_centered = edges_ranked - edge_mean
        denom = edge_std.flatten() * lb_std * n_subj
        denom = np.maximum(denom, 1e-12)
        r_all = np.dot(edge_centered, lb_centered) / denom
        t_stat = r_all * np.sqrt(n_subj - 2) / np.sqrt(np.maximum(1 - r_all**2, 1e-12))
        p_all = 2 * t_dist.sf(np.abs(t_stat), n_subj - 2)
        r_matrix = r_all.reshape(num_nodes, num_nodes)
        p_matrix = p_all.reshape(num_nodes, num_nodes)

    else:
        raise ValueError(f"Unsupported measurement: {measurement}")

    p_mask = (p_matrix < 0.05).astype(float)
    # (roi, roi, n_subj) * (roi, roi, 1); 勿用 p_mask[np.newaxis,np.newaxis,:] 会得到 (1,1,roi,roi)
    train_edges_thed = edges * p_mask[:, :, np.newaxis]
    train_picked_edges = rearrange_edges(train_edges_thed, num_nodes, n_subj)
    return train_picked_edges
