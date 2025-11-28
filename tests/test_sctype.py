"""Tests for sctypepy - ScType cell type annotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from sctypepy import (
    auto_detect_tissue_type,
    calculate_scores,
    get_available_tissues,
    prepare_gene_sets,
    run_sctype,
)
from sctypepy.io import _load_database


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_adata():
    """Create a simple AnnData object with distinct cell type patterns."""
    np.random.seed(42)
    n_cells, n_genes = 100, 50

    gene_names = [
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD8B",  # T cell markers
        "CD19",
        "CD20",
        "MS4A1",
        "CD79A",
        "CD79B",  # B cell markers
        "CD14",
        "CD68",
        "LYZ",
        "CST3",
        "S100A8",  # Monocyte markers
        "GNLY",
        "NKG7",
        "GZMB",
        "GZMA",
        "FCGR3A",  # NK cell markers
    ] + [f"GENE{i}" for i in range(30)]

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    # Create cell type patterns
    X[0:25, 0:5] += 3.0  # T cells
    X[25:50, 5:10] += 3.0  # B cells
    X[50:75, 10:15] += 3.0  # Monocytes
    X[75:100, 15:20] += 3.0  # NK cells

    return sc.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "leiden": pd.Categorical(
                    ["0"] * 25 + ["1"] * 25 + ["2"] * 25 + ["3"] * 25
                )
            },
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )


@pytest.fixture
def simple_gene_sets():
    """Create simple gene sets for testing."""
    gs = {
        "T cells": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
        "B cells": ["CD19", "CD20", "MS4A1", "CD79A", "CD79B"],
        "Monocytes": ["CD14", "CD68", "LYZ", "CST3", "S100A8"],
        "NK cells": ["GNLY", "NKG7", "GZMB", "GZMA", "FCGR3A"],
    }
    gs2 = {
        "T cells": ["CD19", "CD14"],
        "B cells": ["CD3D", "CD14"],
        "Monocytes": ["CD3D", "CD19"],
        "NK cells": ["CD3D", "CD19", "CD14"],
    }
    return gs, gs2


# ============================================================================
# IO Module Tests
# ============================================================================


class TestIO:
    """Tests for IO functions."""

    def test_load_database(self):
        """Database loads correctly with required structure."""
        df = _load_database()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ["tissueType", "cellName", "geneSymbolmore1", "geneSymbolmore2"]:
            assert col in df.columns

    def test_prepare_gene_sets(self):
        """Gene sets are prepared correctly with valid gene symbols."""
        gs, gs2 = prepare_gene_sets("Immune system")

        assert len(gs) > 0
        assert "Macrophages" in gs

        # Check gene quality
        for genes in gs.values():
            for gene in genes:
                assert gene.isupper()
                assert " " not in gene
                assert gene.lower() != "nan"

    def test_prepare_gene_sets_invalid_tissue(self):
        """Invalid tissue type raises ValueError."""
        with pytest.raises(ValueError, match="No cell types found"):
            prepare_gene_sets("NotARealTissue")

    def test_prepare_gene_sets_custom_db(self):
        """Works with custom marker database."""
        custom_db = pd.DataFrame(
            {
                "tissueType": ["Custom", "Custom"],
                "cellName": ["Type A", "Type B"],
                "geneSymbolmore1": ["GENE1,GENE2", "GENE3"],
                "geneSymbolmore2": ["GENE5", ""],
            }
        )
        gs, gs2 = prepare_gene_sets("Custom", db=custom_db)
        assert gs["Type A"] == ["GENE1", "GENE2"]
        assert gs2["Type B"] == []

    def test_get_available_tissues(self):
        """Available tissues are listed correctly."""
        tissues = get_available_tissues()
        assert isinstance(tissues, list)
        assert "Immune system" in tissues
        assert tissues == sorted(tissues)


# ============================================================================
# Scoring Module Tests
# ============================================================================


class TestCalculateScores:
    """Tests for calculate_scores function."""

    def test_basic_output(self, simple_adata, simple_gene_sets):
        """Score output has correct structure."""
        gs, gs2 = simple_gene_sets
        scores = calculate_scores(simple_adata, gs, gs2)

        assert isinstance(scores, pd.DataFrame)
        assert scores.shape == (simple_adata.n_obs, len(gs))
        assert list(scores.index) == list(simple_adata.obs_names)
        assert set(scores.columns) == set(gs.keys())

    def test_sparse_dense_equivalence(self, simple_adata, simple_gene_sets):
        """Sparse and dense matrices give equivalent results."""
        gs, gs2 = simple_gene_sets
        sparse_adata = simple_adata.copy()
        sparse_adata.X = sparse.csr_matrix(sparse_adata.X)

        scores_dense = calculate_scores(simple_adata, gs, gs2)
        scores_sparse = calculate_scores(sparse_adata, gs, gs2)

        np.testing.assert_allclose(scores_dense.values, scores_sparse.values, rtol=1e-4)

    def test_options(self, simple_adata):
        """Various scoring options work correctly."""
        gs = {"T cells": ["CD3D", "CD3E", "CD4"]}

        # Without negative markers
        scores = calculate_scores(simple_adata, gs, gs2=None)
        assert isinstance(scores, pd.DataFrame)

        # Without scaling
        scores = calculate_scores(simple_adata, gs, scale=False)
        assert isinstance(scores, pd.DataFrame)

        # Case-insensitive gene matching
        gs_lower = {"T cells": ["cd3d", "cd3e"]}
        scores = calculate_scores(simple_adata, gs_lower, gene_names_to_uppercase=True)
        assert not scores["T cells"].isna().all()

        # Missing genes handled gracefully
        gs_missing = {"Fake": ["NOTAREALGENE"]}
        scores = calculate_scores(simple_adata, gs_missing)
        assert isinstance(scores, pd.DataFrame)

    def test_correct_cell_scoring(self, simple_adata, simple_gene_sets):
        """Cells score highest for their matching cell type."""
        gs, gs2 = simple_gene_sets
        scores = calculate_scores(simple_adata, gs, gs2)

        # T cells (0-24) should score highest for "T cells"
        t_cell_scores = scores.iloc[0:25]["T cells"].mean()
        other_scores = scores.iloc[0:25].drop(columns="T cells").mean().mean()
        assert t_cell_scores > other_scores

    def test_marker_sensitivity(self):
        """Unique markers contribute more to scores than shared markers."""
        np.random.seed(42)
        gs = {
            "Type A": ["SHARED1", "SHARED2", "UNIQUE_A"],
            "Type B": ["SHARED1", "SHARED2", "UNIQUE_B"],
        }

        X = np.random.rand(20, 4).astype(np.float32) * 0.5
        X[0:10, 2] = 5.0  # High UNIQUE_A in first 10 cells

        adata = sc.AnnData(
            X=X,
            obs=pd.DataFrame({"c": ["0"] * 20}, index=[f"c{i}" for i in range(20)]),
            var=pd.DataFrame(index=["SHARED1", "SHARED2", "UNIQUE_A", "UNIQUE_B"]),
        )

        scores = calculate_scores(adata, gs)
        assert scores["Type A"].iloc[0:10].mean() > scores["Type B"].iloc[0:10].mean()


# ============================================================================
# Core Module Tests
# ============================================================================


class TestAutoDetect:
    """Tests for auto_detect_tissue_type function."""

    def test_basic_output(self, simple_adata):
        """Returns sorted DataFrame with tissue scores."""
        result = auto_detect_tissue_type(simple_adata, groupby="leiden", verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert {"tissue", "score"}.issubset(result.columns)
        scores = result["score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_invalid_groupby(self, simple_adata):
        """Invalid groupby raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            auto_detect_tissue_type(simple_adata, groupby="nonexistent", verbose=False)


class TestRunSctype:
    """Tests for run_sctype main function."""

    def test_basic_output(self, simple_adata):
        """run_sctype produces correct output structure."""
        adata = run_sctype(simple_adata, tissue_type="Immune system", groupby="leiden")

        assert isinstance(adata, sc.AnnData)
        assert "sctype_classification" in adata.obs.columns
        assert "sctype_scores" in adata.obsm
        assert isinstance(adata.obsm["sctype_scores"], pd.DataFrame)
        assert not adata.obs["sctype_classification"].isna().any()

    def test_error_handling(self, simple_adata):
        """Invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError, match="not found"):
            run_sctype(simple_adata, tissue_type="Immune system", groupby="bad_col")

        with pytest.raises(ValueError, match="No cell types found"):
            run_sctype(simple_adata, tissue_type="FakeTissue", groupby="leiden")

    def test_classification_validity(self, simple_adata):
        """Classifications are valid cell types or 'Unknown'."""
        adata = run_sctype(simple_adata, tissue_type="Immune system", groupby="leiden")
        gs, _ = prepare_gene_sets("Immune system")
        valid = set(gs.keys()) | {"Unknown"}

        for cls in adata.obs["sctype_classification"].unique():
            assert cls in valid

    def test_sparse_matrix(self, simple_adata):
        """Works with sparse matrices."""
        sparse_adata = simple_adata.copy()
        sparse_adata.X = sparse.csr_matrix(sparse_adata.X)
        adata = run_sctype(sparse_adata, tissue_type="Immune system", groupby="leiden")
        assert "sctype_classification" in adata.obs.columns

    def test_auto_detect_tissue(self, simple_adata):
        """Auto-detects tissue type when not specified."""
        adata = run_sctype(simple_adata, tissue_type=None, groupby="leiden")
        assert "sctype_classification" in adata.obs.columns

    def test_custom_database(self, simple_adata):
        """Works with custom marker database."""
        custom_db = pd.DataFrame(
            {
                "tissueType": ["Custom"] * 4,
                "cellName": ["T cells", "B cells", "Monocytes", "NK cells"],
                "geneSymbolmore1": [
                    "CD3D,CD3E,CD4",
                    "CD19,CD20,MS4A1",
                    "CD14,CD68,LYZ",
                    "GNLY,NKG7,GZMB",
                ],
                "geneSymbolmore2": ["", "", "", ""],
            }
        )
        adata = run_sctype(
            simple_adata, tissue_type="Custom", groupby="leiden", db=custom_db
        )
        assert "sctype_classification" in adata.obs.columns


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_minimal_data(self):
        """Works with minimal data (single cell, single cluster)."""
        for n_cells in [1, 50]:
            adata = sc.AnnData(
                X=np.random.rand(n_cells, 10).astype(np.float32),
                obs=pd.DataFrame(
                    {"c": pd.Categorical(["0"] * n_cells)},
                    index=[f"c{i}" for i in range(n_cells)],
                ),
                var=pd.DataFrame(index=[f"G{i}" for i in range(10)]),
            )
            result = run_sctype(adata, tissue_type="Immune system", groupby="c")
            assert "sctype_classification" in result.obs.columns

    def test_gene_case_handling(self):
        """Handles various gene name cases correctly."""
        np.random.seed(42)
        gs = {"T cells": ["CD3D", "CD3E"]}

        # Lowercase genes in adata
        adata = sc.AnnData(
            X=np.random.rand(10, 2).astype(np.float32),
            obs=pd.DataFrame({"c": ["0"] * 10}, index=[f"c{i}" for i in range(10)]),
            var=pd.DataFrame(index=["cd3d", "cd3e"]),
        )
        scores = calculate_scores(adata, gs, gene_names_to_uppercase=True)
        assert isinstance(scores, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
