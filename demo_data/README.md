# 10x Visium BRCA dataset

`10xVisiumBRCA.h5ad` contains the same 2,518 spots, 334 genes, and two pathways
as the original BRCA demo.

- `X` and `layers["counts"]`: sparse raw counts.
- `obs`: spot barcodes, array row/column, and library ID.
- `var`: gene symbols and feature types.
- `obsm["spatial"]`: x/y coordinates (array column, array row).
- `uns["spatial"]["BRCA"]`: spatial metadata.
- `varm["pathway_membership"]`: gene-by-pathway membership table.

Coordinates are array indices. Tissue images, pixel coordinates, and image scale
factors are not included. Source-file checksums are stored in `uns["dataset"]`.
