from __future__ import annotations

from importlib import resources
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _load_database() -> pd.DataFrame:
    """Load the built-in marker database."""
    with resources.files("sctypepy.data").joinpath("markers.csv").open("r") as f:
        return pd.read_csv(f)


def get_available_tissues(db: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Get available tissue types in the database.

    Parameters
    ----------
    db : pd.DataFrame, optional
        Custom marker database.

    Returns
    -------
    List[str]
        Sorted list of tissue types.
    """
    if db is None:
        db = _load_database()
    return sorted(db["tissueType"].unique().tolist())


def prepare_gene_sets(
    tissue: str,
    db: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Prepare positive and negative gene sets for a tissue type.

    Parameters
    ----------
    tissue : str
        Tissue type to filter by.
    db : pd.DataFrame, optional
        Custom marker database with columns:
        'tissueType', 'cellName', 'geneSymbolmore1', 'geneSymbolmore2'.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        (positive_markers, negative_markers) dictionaries.
    """
    df = _load_database() if db is None else db
    cell_markers = df[df["tissueType"] == tissue]

    if len(cell_markers) == 0:
        raise ValueError(f"No cell types found for tissue: {tissue}")

    def parse_genes(val) -> List[str]:
        s = str(val).replace(" ", "")
        if s.lower() == "nan" or not s:
            return []
        return [m.upper() for m in s.split(",") if m and m.lower() != "nan"]

    gs = {
        row["cellName"]: parse_genes(row["geneSymbolmore1"])
        for _, row in cell_markers.iterrows()
    }
    gs2 = {
        row["cellName"]: parse_genes(row["geneSymbolmore2"])
        for _, row in cell_markers.iterrows()
    }

    return gs, gs2
