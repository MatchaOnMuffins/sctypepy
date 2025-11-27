from __future__ import annotations

from importlib import resources
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def _load_database() -> pd.DataFrame:
    """
    Load the marker database from the package resources.
    """
    with resources.files("sctypepy.data").joinpath("markers.csv").open("r") as f:
        return pd.read_csv(f)


def get_available_tissues(db: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Get a list of available tissue types in the database.

    Parameters
    ----------
    db : pd.DataFrame, optional
        Custom marker database. If None, uses the built-in database.

    Returns
    -------
    List[str]
        List of available tissue types.
    """
    if db is None:
        db = _load_database()
    return sorted(db["tissueType"].unique().tolist())


def prepare_gene_sets(
    tissue: str,
    db: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Prepare gene sets from the database file for a specific tissue type.

    Parameters
    ----------
    tissue : str
        Tissue type to filter by (e.g. 'Immune system', 'Pancreas').
    db : pd.DataFrame, optional
        Custom marker database DataFrame. Must have columns:
        'tissueType', 'cellName', 'geneSymbolmore1', 'geneSymbolmore2'.
        If None, uses the built-in database.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        Two dictionaries: (positive_markers, negative_markers).
        Keys are cell types, values are lists of gene symbols.
    """
    if db is None:
        df = _load_database()
    else:
        df = db

    # Filter by tissue type
    cell_markers = df[df["tissueType"] == tissue].copy()

    if len(cell_markers) == 0:
        raise ValueError(f"No cell types found for tissue: {tissue}")

    gs = {}
    gs2 = {}

    for _, row in cell_markers.iterrows():
        cell_name = row["cellName"]

        # Helper to parse gene lists
        def parse_genes(val):
            s = str(val).replace(" ", "")
            if s.lower() == "nan" or not s:
                return []
            return [m.upper() for m in s.split(",") if m and m.lower() != "nan"]

        gs[cell_name] = parse_genes(row["geneSymbolmore1"])
        gs2[cell_name] = parse_genes(row["geneSymbolmore2"])

    return gs, gs2

