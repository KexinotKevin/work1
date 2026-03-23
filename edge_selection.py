import math

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _infer_num_nodes(num_edges):
    num_nodes = (1 + math.sqrt(1 + 8 * num_edges)) / 2
    if not float(num_nodes).is_integer():
        raise ValueError(
            f"Cannot infer node count from {num_edges} edges. "
            "Expected a triangular-number edge vector."
        )
    return int(num_nodes)


def haufe_transform(train_picked_edges, lb_col, model_coef, num_nodes=None):
    cov_x = np.cov(np.transpose(train_picked_edges.T))
    cov_y = np.cov(lb_col)
    haufe_weights = np.matmul(cov_x, model_coef) * (1 / cov_y)
    if num_nodes is None:
        num_nodes = _infer_num_nodes(haufe_weights.shape[0])
    haufe_all_weights = reshape_feat_to_net(haufe_weights, num_nodes)
    return haufe_all_weights


def rearrange_edges(graphs, num_nodes, num_subjs):
    rows, cols = np.triu_indices(num_nodes, k=1)
    all_edges = np.empty((len(rows), num_subjs))
    for i in range(num_subjs):
        all_edges[:, i] = graphs[rows, cols, i]
    return all_edges


def reshape_feat_to_net(feature_vector, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes))
    triu_indices = np.triu_indices(num_nodes, k=1)
    matrix[triu_indices] = feature_vector
    matrix = matrix + matrix.T
    return matrix


def select_sig_edges(lb_list, edges, num_nodes, measurement="pcorr"):
    p_mask = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            conn = edges[i, j, :]
            if measurement == "pcorr":
                pcorr = pearsonr(conn, lb_list)
            elif measurement == "rankcorr":
                pcorr = spearmanr(conn, lb_list)
            else:
                raise ValueError(f"Unsupported measurement: {measurement}")
            pcorr_res[i, j, 0] = pcorr.statistic
            pcorr_res[i, j, 1] = pcorr.pvalue
            if pcorr.pvalue < 0.05:
                p_mask[i, j] = 1

    train_edges_thed = np.zeros_like(edges)
    for ss in range(edges.shape[2]):
        train_edges_thed[:, :, ss] = edges[:, :, ss] * p_mask
    train_picked_edges = rearrange_edges(train_edges_thed, num_nodes, edges.shape[2])
    return train_picked_edges
