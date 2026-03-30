# h5ad-inspect

CLI tool for quick inspection of [AnnData](https://anndata.readthedocs.io/) `.h5ad` files.

## Usage

```
h5ad-inspect <filename> <section>
```

Prints a sorted, newline-separated list for the given section.

| Section | Output |
|---------|--------|
| `obs` | obs column names (cell metadata) |
| `var` | var column names (gene metadata) |
| `obs_index` | obs index values (cell barcodes / names) |
| `var_index` | var index values (gene names) |
| `obsm` | obsm keys (embeddings, e.g. `X_pca`, `X_umap`) |
| `layers` | layer names |
| `uns` | uns keys (unstructured metadata) |

## Examples

```sh
h5ad-inspect data.h5ad obs
# batch
# cell_type
# n_genes

h5ad-inspect data.h5ad var_index
# ACTB
# GAPDH
# ...

h5ad-inspect data.h5ad obsm
# X_pca
# X_umap
```

## Build

Requires the HDF5 C library.

```sh
# with Nix
nix build

# manually (set PKG_CONFIG_PATH to your HDF5 installation)
PKG_CONFIG_PATH=/path/to/hdf5/lib/pkgconfig cargo build --release
```
