import scanpy as sc
from sctypepy import run_sctype

adata = sc.datasets.pbmc3k()

sc.pp.neighbors(adata)
sc.tl.leiden(adata)

adata = run_sctype(adata, tissue_type="Immune system", groupby="leiden")

print(adata.obs["sctype_classification"])
