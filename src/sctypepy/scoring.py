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
    Calculate ScType scores for each cell efficiently.

    Optimized to work with sparse matrices without densifying the entire dataset.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gs : dict
        Dictionary of positive gene sets {cell_type: [genes]}.
    gs2 : dict, optional
        Dictionary of negative gene sets {cell_type: [genes]}.
    scale : bool
        Whether to Z-score the data implicitly (Recommended).
    gene_names_to_uppercase : bool
        Whether to convert adata gene names to uppercase for matching.

    Returns
    -------
    pd.DataFrame
        DataFrame of scores (n_cells x n_cell_types).
    """
    # 1. Setup Gene Mapping
    if gene_names_to_uppercase:
        var_names = adata.var_names.str.upper()
    else:
        var_names = adata.var_names

    # Create a map from gene name to index (handle duplicates by taking the first/last match?)
    # Using a series for fast lookup
    gene_to_idx = pd.Series(np.arange(len(var_names)), index=var_names)
    # Warning: if there are duplicate gene names, this might pick one arbitrarily.
    # Ideally, adata shouldn't have duplicate var_names.
    if not var_names.is_unique:
        warnings.warn(
            "Gene names are not unique. ScType scoring might be inaccurate for duplicated genes."
        )
        # Remove duplicates from the index to avoid errors, keeping first
        gene_to_idx = gene_to_idx[~gene_to_idx.index.duplicated(keep="first")]

    # 2. Marker Sensitivity
    # Count frequency of each gene across all POSITIVE cell types
    # Genes appearing in fewer cell types are MORE informative (higher sensitivity)
    # R: scales::rescale(marker_stat, to = c(0,1), from = c(length(gs),1))
    # This means: count=length(gs) -> 0, count=1 -> 1
    all_markers = [g for markers in gs.values() for g in markers]
    marker_counts = pd.Series(all_markers).value_counts()

    n_cell_types = len(gs)
    # Rescale from [n_cell_types, 1] to [0, 1]
    # sensitivity = (count - n_cell_types) / (1 - n_cell_types)
    #             = (n_cell_types - count) / (n_cell_types - 1)
    if n_cell_types > 1:
        marker_sensitivity = (n_cell_types - marker_counts) / (n_cell_types - 1)
    else:
        # Only one cell type, all markers have sensitivity 1
        marker_sensitivity = marker_counts * 0 + 1

    # 3. Precompute Statistics (Mean, Std) for Implicit Scaling
    # We only need stats for genes that are in our marker lists
    # But calculating for all is usually fast enough or we can filter.
    # To keep it simple and vectorized, we'll compute for all genes.

    X = adata.X

    if scale:
        if sparse.issparse(X):
            # Calculate mean and var without densifying
            mean = np.array(X.mean(axis=0)).flatten()
            # var = E[X^2] - (E[X])^2
            # Use a small chunk to estimate or just calculate properly?
            # For sparse, X.multiply(X) is efficient
            mean_sq = np.array(X.multiply(X).mean(axis=0)).flatten()
            var = mean_sq - mean**2
            # fix precision issues
            var[var < 0] = 0
            std = np.sqrt(var)
        else:
            mean = np.array(X.mean(axis=0)).flatten()
            std = np.array(X.std(axis=0)).flatten()

        # Avoid division by zero for constant genes
        std[std == 0] = 1e-12  # Technically they should be 0 influence, handled by weights below
        inv_std = 1.0 / std
        # If std was 0, gene is constant. Z-score should be 0.
        # With inv_std, we get huge numbers. We must mask these out.
        inv_std[std < 1e-9] = 0
    else:
        # If no scaling, effective mean=0, std=1 for the formula logic (identity transform)
        mean = np.zeros(X.shape[1])
        inv_std = np.ones(X.shape[1])

    # 4. Score Calculation (Vectorized)
    # Score = (X @ Weights) - Offsets
    # Weights matrix: (n_genes, n_celltypes)
    # Offsets vector: (n_celltypes,)

    cell_types = list(gs.keys())
    n_types = len(cell_types)
    n_genes = X.shape[1]

    # We'll build the Weights matrix using sparse coordinate format
    # Rows: genes, Cols: cell_types

    def compute_term(gene_sets_dict, sign=1.0):
        w_rows = []
        w_cols = []
        w_data = []
        offsets = np.zeros(n_types)

        for i, ct in enumerate(cell_types):
            if ct not in gene_sets_dict:
                continue

            genes = gene_sets_dict[ct]
            # Filter genes present in data
            valid_genes = [g for g in genes if g in gene_to_idx.index]

            if not valid_genes:
                continue

            indices = gene_to_idx[valid_genes].values

            # Get sensitivities (default to 1.0 if not in sensitivity map, though they should be for POS)
            sens = np.array([marker_sensitivity.get(g, 1.0) for g in valid_genes])

            # Weights for these genes in this cell type
            # Weight = sens / std
            # We normalize by sqrt(N) per cell type logic here
            norm_factor = np.sqrt(len(valid_genes))

            current_inv_std = inv_std[indices]
            current_mean = mean[indices]

            # w_g = sens_g * inv_std_g / sqrt(N)
            w = (sens * current_inv_std) / norm_factor

            # offset = sum(mean_g * w_g)
            offset = np.sum(current_mean * w)

            w_rows.extend(indices)
            w_cols.extend([i] * len(indices))
            w_data.extend(w)
            offsets[i] = offset

        # Construct sparse weight matrix
        W = sparse.csc_matrix((w_data, (w_rows, w_cols)), shape=(n_genes, n_types))
        return W, offsets

    # Positive scores
    W_pos, Offsets_pos = compute_term(gs)

    # Negative scores
    if gs2:
        W_neg, Offsets_neg = compute_term(gs2)
    else:
        W_neg, Offsets_neg = None, None

    # Compute final scores
    # X is (n_cells, n_genes)
    # W is (n_genes, n_types)
    # Result X @ W is (n_cells, n_types)

    # Convert X to csr if it's not for fast multiplication
    if sparse.issparse(X) and not sparse.isspmatrix_csr(X):
        X_calc = X.tocsr()
    else:
        X_calc = X

    Scores_pos = X_calc @ W_pos
    # Subtract offsets (broadcasting)
    Scores_pos = Scores_pos - Offsets_pos

    if W_neg is not None:
        Scores_neg = X_calc @ W_neg
        Scores_neg = Scores_neg - Offsets_neg
        Final_Scores = Scores_pos - Scores_neg
    else:
        Final_Scores = Scores_pos

    return pd.DataFrame(Final_Scores, index=adata.obs_names, columns=cell_types)

