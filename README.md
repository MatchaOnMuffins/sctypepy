# ScTypePy: Python implementation of ScType

A Python implementation of [ScType](https://github.com/IanevskiAleksandr/sc-type) for automatic cell type annotation of single-cell RNA-seq data.

ScType is described in the following publication:  
**Nature Communications (2022):** https://doi.org/10.1038/s41467-022-28803-w

## Installation

```bash
pip install sctypepy
```

## Quickstart

```python
import scanpy as sc
from sctypepy import run_sctype

# Load and preprocess your data
adata = sc.datasets.pbmc3k()
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

adata = run_sctype(adata, tissue_type="Immune system", groupby="leiden")

print(adata.obs["sctype_classification"].value_counts())
```

## Example
Example usages can be found in the [example](example/) directory.

## Usage

### Auto-detect tissue type

If you're unsure which tissue type to use:

```python
from sctypepy import auto_detect_tissue_type

tissue_df = auto_detect_tissue_type(adata, groupby="leiden")
print(tissue_df.head())
```

### Available tissue types

```python
from sctypepy import get_available_tissues

print(get_available_tissues())
```

### Custom marker database

Provide your own markers as a DataFrame with columns: `tissueType`, `cellName`, `geneSymbolmore1`, `geneSymbolmore2`.

```python
import pandas as pd

custom_db = pd.DataFrame({
    "tissueType": ["Brain", "Brain"],
    "cellName": ["Neuron", "Astrocyte"],
    "geneSymbolmore1": ["SNAP25,SYT1,RBFOX3", "GFAP,AQP4,S100B"],
    "geneSymbolmore2": ["", ""],
})

adata = run_sctype(adata, tissue_type="Brain", db=custom_db)
```

## Output

After running `run_sctype()`:

- `adata.obs["sctype_classification"]` — predicted cell type per cell
- `adata.obsm["sctype_scores"]` — raw scores for each cell type


## Authors & Contributions

- **Original ScType algorithm and R implementation:** Aleksandr Ianevski and contributors  
  https://github.com/IanevskiAleksandr/sc-type
