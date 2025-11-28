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
    Auto-detect the best matching tissue type from the marker database.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column in adata.obs for clustering.
    db : pd.DataFrame, optional
        Custom marker database.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Ranked tissue types with scores.
    """
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    results = []
    for tissue in get_available_tissues(db):
        if verbose:
            print(f"Checking... {tissue}")

        try:
            gs, gs2 = prepare_gene_sets(tissue, db=db)
        except ValueError:
            continue

        sc_scores = calculate_scores(adata, gs, gs2)
        cluster_scores = sc_scores.groupby(adata.obs[groupby], observed=True).sum()
        mean_score = cluster_scores.max(axis=1).mean()
        results.append({"tissue": tissue, "score": mean_score})

    result_df = (
        pd.DataFrame(results)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    if verbose and len(result_df) > 0:
        print(
            f"\nTop tissue type: {result_df.iloc[0]['tissue']} (score: {result_df.iloc[0]['score']:.2f})"
        )

    return result_df


def run_sctype(
    adata: sc.AnnData,
    tissue_type: Optional[str] = None,
    groupby: str = "leiden",
    db: Optional[pd.DataFrame] = None,
) -> sc.AnnData:
    """
    Run ScType cell type annotation on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    tissue_type : str, optional
        Tissue type for marker lookup. Auto-detected if None.
    groupby : str
        Column in adata.obs for clustering.
    db : pd.DataFrame, optional
        Custom marker database.

    Returns
    -------
    AnnData
        Modified AnnData with 'sctype_classification' in obs.
    """
    if tissue_type is None:
        print("Tissue type not specified. Auto-detecting...")
        tissue_df = auto_detect_tissue_type(adata, groupby=groupby, db=db, verbose=True)
        if len(tissue_df) == 0:
            raise ValueError(
                "Could not auto-detect tissue type. Please specify tissue_type."
            )
        tissue_type = tissue_df.iloc[0]["tissue"]
        print(f"Using detected tissue type: {tissue_type}")

    print(f"Preparing gene sets for tissue: {tissue_type}...")
    gs, gs2 = prepare_gene_sets(tissue_type, db=db)

    print("Calculating scores...")
    sc_scores = calculate_scores(adata, gs, gs2)
    adata.obsm["sctype_scores"] = sc_scores

    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    print(f"Aggregating scores by cluster column: '{groupby}'...")
    cluster_scores = sc_scores.groupby(adata.obs[groupby], observed=True).sum()
    cluster_sizes = adata.obs[groupby].value_counts()

    predictions = {}
    for cluster in cluster_scores.index:
        scores = cluster_scores.loc[cluster].sort_values(ascending=False)
        if scores.empty:
            continue

        top_type, top_score = scores.index[0], scores.iloc[0]
        n_cells = cluster_sizes.get(cluster, 1)

        # Low confidence threshold: avg z-score < 0.25
        predictions[cluster] = "Unknown" if top_score < (n_cells / 4) else top_type

    adata.obs["sctype_classification"] = adata.obs[groupby].map(predictions)

    print("Done! Added 'sctype_classification' to adata.obs")
    return adata
