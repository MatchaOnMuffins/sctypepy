from __future__ import annotations

from typing import Optional

import pandas as pd
import scanpy as sc

from .io import get_available_tissues, prepare_gene_sets
from .scoring import calculate_scores


def auto_detect_tissue_type(
    adata: sc.AnnData,
    groupby: str = "leiden",
    db: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Automatically detect the tissue type of the dataset.

    Scores the dataset against all available tissue types and returns
    a ranked list of tissue types by their average cluster scores.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    groupby : str
        The column in adata.obs to group cells by for scoring.
    db : pd.DataFrame, optional
        Custom marker database. If None, uses the built-in database.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'tissue' and 'score', sorted by score descending.
        The first row is the most likely tissue type.
    """
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    tissues = get_available_tissues(db)
    results = []

    for tissue in tissues:
        if verbose:
            print(f"Checking... {tissue}")

        try:
            gs, gs2 = prepare_gene_sets(tissue, db=db)
        except ValueError:
            continue

        sc_scores = calculate_scores(adata, gs, gs2)

        # Aggregate by cluster
        cluster_scores = sc_scores.groupby(adata.obs[groupby], observed=True).sum()

        # Get top score per cluster
        top_scores = cluster_scores.max(axis=1)

        # Mean of top scores across clusters
        mean_score = top_scores.mean()
        results.append({"tissue": tissue, "score": mean_score})

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

    if verbose and len(result_df) > 0:
        print(f"\nTop tissue type: {result_df.iloc[0]['tissue']} (score: {result_df.iloc[0]['score']:.2f})")

    return result_df


def run_sctype(
    adata: sc.AnnData,
    tissue_type: Optional[str] = None,
    groupby: str = "leiden",
    db: Optional[pd.DataFrame] = None,
) -> sc.AnnData:
    """
    Main wrapper function to run ScType on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    tissue_type : str, optional
        The tissue type to search in the database.
        If None, auto-detects the tissue type.
    groupby : str
        The column in adata.obs to group cells by for classification.
    db : pd.DataFrame, optional
        Custom marker database. If None, uses the built-in database.

    Returns
    -------
    AnnData
        The modified AnnData object with 'sctype_classification' in obs.
    """
    # Auto-detect tissue type if not provided
    if tissue_type is None:
        print("Tissue type not specified. Auto-detecting...")
        tissue_df = auto_detect_tissue_type(adata, groupby=groupby, db=db, verbose=True)
        if len(tissue_df) == 0:
            raise ValueError("Could not auto-detect tissue type. Please specify tissue_type.")
        tissue_type = tissue_df.iloc[0]["tissue"]
        print(f"Using detected tissue type: {tissue_type}")

    print(f"Preparing gene sets for tissue: {tissue_type}...")
    gs, gs2 = prepare_gene_sets(tissue_type, db=db)

    print("Calculating scores...")
    sc_scores = calculate_scores(adata, gs, gs2)

    # Store scores in adata for reference
    adata.obsm["sctype_scores"] = sc_scores

    print(f"Aggregating scores by cluster column: '{groupby}'...")
    # Aggregate by cluster (Sum)
    # We need to ensure 'groupby' column is categorical or manageable
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    # Fast aggregation using pandas
    # We want sum of scores per cluster
    # One-hot encode clusters? Or just groupby sum
    cluster_scores = sc_scores.groupby(adata.obs[groupby], observed=True).sum()

    predictions = {}
    # Calculate cluster sizes for thresholding
    cluster_sizes = adata.obs[groupby].value_counts()

    for cluster in cluster_scores.index:
        scores = cluster_scores.loc[cluster].sort_values(ascending=False)
        if scores.empty:
            continue

        top_type = scores.index[0]
        top_score = scores.iloc[0]

        n_cells = cluster_sizes.get(cluster, 1)

        # Low confidence check (score < n_cells/4 implies avg z-score < 0.25)
        if top_score < (n_cells / 4):
            predictions[cluster] = "Unknown"
        else:
            predictions[cluster] = top_type

    adata.obs["sctype_classification"] = adata.obs[groupby].map(predictions)

    print("Done! Added 'sctype_classification' to adata.obs")
    return adata

