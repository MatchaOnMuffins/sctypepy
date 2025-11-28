from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def calculate_scores(
    adata: sc.AnnData,
    gs: Dict[str, List[str]],
    gs2: Optional[Dict[str, List[str]]] = None,
    scale: bool = True,
    gene_names_to_uppercase: bool = True,
) -> pd.DataFrame:
    """
    Calculate ScType scores for each cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gs : dict
        Positive gene sets {cell_type: [genes]}.
    gs2 : dict, optional
        Negative gene sets {cell_type: [genes]}.
    scale : bool
        Whether to Z-score normalize the data.
    gene_names_to_uppercase : bool
        Whether to convert gene names to uppercase for matching.

    Returns
    -------
    pd.DataFrame
        Scores matrix (n_cells x n_cell_types).
    """
    var_names = adata.var_names.str.upper() if gene_names_to_uppercase else adata.var_names
    gene_to_idx = pd.Series(np.arange(len(var_names)), index=var_names)

    if not var_names.is_unique:
        warnings.warn(
            "Gene names are not unique. ScType scoring might be inaccurate for duplicated genes."
        )
        gene_to_idx = gene_to_idx[~gene_to_idx.index.duplicated(keep="first")]

    all_markers = [g for markers in gs.values() for g in markers]
    marker_counts = pd.Series(all_markers).value_counts()

    n_cell_types = len(gs)
    # Marker sensitivity: genes in fewer cell types are more informative
    # Rescaled from [n_cell_types, 1] to [0, 1]
    if n_cell_types > 1:
        marker_sensitivity = (n_cell_types - marker_counts) / (n_cell_types - 1)
    else:
        marker_sensitivity = marker_counts * 0 + 1

    X = adata.X
    mean, inv_std = _compute_scaling_params(X, scale)

    cell_types = list(gs.keys())
    n_types = len(cell_types)
    n_genes = X.shape[1]

    def compute_term(gene_sets_dict):
        w_rows, w_cols, w_data = [], [], []
        offsets = np.zeros(n_types)

        for i, ct in enumerate(cell_types):
            if ct not in gene_sets_dict:
                continue

            valid_genes = [g for g in gene_sets_dict[ct] if g in gene_to_idx.index]
            if not valid_genes:
                continue

            indices = gene_to_idx[valid_genes].values
            sens = np.array([marker_sensitivity.get(g, 1.0) for g in valid_genes])
            norm_factor = np.sqrt(len(valid_genes))

            current_inv_std = inv_std[indices]
            current_mean = mean[indices]

            w = (sens * current_inv_std) / norm_factor
            offset = np.sum(current_mean * w)

            w_rows.extend(indices)
            w_cols.extend([i] * len(indices))
            w_data.extend(w)
            offsets[i] = offset

        W = sparse.csc_matrix((w_data, (w_rows, w_cols)), shape=(n_genes, n_types))
        return W, offsets

    W_pos, offsets_pos = compute_term(gs)
    W_neg, offsets_neg = compute_term(gs2) if gs2 else (None, None)

    X_calc = X.tocsr() if sparse.issparse(X) and not sparse.isspmatrix_csr(X) else X

    scores = (X_calc @ W_pos) - offsets_pos
    if W_neg is not None:
        scores = scores - ((X_calc @ W_neg) - offsets_neg)

    return pd.DataFrame(scores, index=adata.obs_names, columns=cell_types)


def _compute_scaling_params(X, scale: bool):
    """Compute mean and inverse std for Z-score normalization."""
    if not scale:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])

    if sparse.issparse(X):
        mean = np.array(X.mean(axis=0)).flatten()
        mean_sq = np.array(X.multiply(X).mean(axis=0)).flatten()
        var = np.maximum(mean_sq - mean**2, 0)
        std = np.sqrt(var)
    else:
        mean = np.array(X.mean(axis=0)).flatten()
        std = np.array(X.std(axis=0)).flatten()

    std[std == 0] = 1e-12
    inv_std = 1.0 / std
    inv_std[std < 1e-9] = 0

    return mean, inv_std
