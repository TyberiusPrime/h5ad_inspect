# h5ad-inspect

A fast command-line tool for inspecting and exporting data from [AnnData](https://anndata.readthedocs.io/) `.h5ad` files.

## Installation

```
cargo install --path .
```

## Usage

```
h5ad-inspect <filename> <section>
h5ad-inspect <filename> export [--binary] [--layer <name>] <subcommand> [<name>]
```

---

## Inspect mode

Lists the available keys/columns in a section of the file.

```bash
h5ad-inspect data.h5ad obs         # list obs column names
h5ad-inspect data.h5ad var         # list var column names
h5ad-inspect data.h5ad uns         # list uns keys
h5ad-inspect data.h5ad obsm        # list obsm keys
h5ad-inspect data.h5ad layers      # list layer names
h5ad-inspect data.h5ad obs_index   # list obs index values (sorted, use 'export obs_index' for original order)
h5ad-inspect data.h5ad var_index   # list var index values (sorted, use 'export var_index' for original order)
h5ad-inspect data.h5ad shape       # print n_obs and n_var (number of cells and genes)
```

---

## Export mode

Exports values to stdout, one value per line.

```
h5ad-inspect <filename> export <subcommand> [<name>]
```

### Subcommands

| Subcommand | Argument | Output |
|---|---|---|
| `obs_index` | — | obs index values, in original order |
| `var_index` | — | var index values, in original order |
| `obs` | column name | values of an obs column |
| `var` | column name | values of a var column |
| `row` | obs ID | X matrix row for that cell (one value per gene) |
| `column` | var ID | X matrix column for that gene (one value per cell) |
| `obssum` | — | sum of X across all vars, one value per obs (row sums) |
| `varsum` | — | sum of X across all obs, one value per var (col sums) |
| `matrix_csr` | — | full X matrix as a NumPy `.npz` stream in CSR (`data`/`indices`/`indptr`/`shape`); see [Full matrix](#full-matrix-csr--csc-numpy-npz) |
| `matrix_csc` | — | full X matrix as a NumPy `.npz` stream in CSC (`data`/`indices`/`indptr`/`shape`); see [Full matrix](#full-matrix-csr--csc-numpy-npz) |
| `matrix_cellranger_v3_hdf5` | output path | full X matrix as a 10x CellRanger v3 `.h5` file, readable by `Seurat::Read10X_h5`; see [10x / Seurat](#10x-cellranger-v3-hdf5-seurat) |
| `obsm` | key | a 2-D obsm embedding (e.g. `X_pca`, `X_umap`) in obs order; one tab-separated line per obs, `n_components` values per line |
| `obs_categories` | column name | order of the categories  of this column |
| `var_categories` | column name | order of the categories  of this column |
| `obs_encoding` | column name | JSON object with the column's h5ad encoding (`categorical`, `bool`, or `numeric`); `categorical` columns also include an in-order `categories` array |
| `var_encoding` | column name | JSON object with the column's h5ad encoding (`categorical`, `bool`, or `numeric`); `categorical` columns also include an in-order `categories` array |
Categorical columns are decoded to their string labels.

```bash
# Export a single obs column
h5ad-inspect data.h5ad export obs cell_type

# Export the X row for a specific cell (one float per gene, newline-separated)
h5ad-inspect data.h5ad export row AAACCTGAGAAGGCCT-1

# Export the X column for a specific gene
h5ad-inspect data.h5ad export column GAPDH

# Sum X across all genes per cell (total counts per cell)
h5ad-inspect data.h5ad export obssum

# Sum X across all cells per gene (total counts per gene)
h5ad-inspect data.h5ad export varsum

# Export a 2-D embedding in obs order (one tab-separated row per cell)
h5ad-inspect data.h5ad export obsm X_umap
```

### Full matrix (CSR / CSC, NumPy `.npz`)

`export matrix_csr` and `export matrix_csc` each stream the **entire X matrix**
to stdout as a NumPy `.npz` archive holding a single sparse representation,
regardless of how X is stored on disk (dense, CSR, or CSC):

`matrix_csr` members:

| Member | dtype | contents |
|---|---|---|
| `csr_data` | float64 | CSR data |
| `csr_indices` | int64 | CSR column indices |
| `csr_indptr` | int64 | CSR row pointers (`n_obs + 1`) |
| `csr_shape` | int64 | `(n_obs, n_var)` |

`matrix_csc` members:

| Member | dtype | contents |
|---|---|---|
| `csc_data` | float64 | CSC data |
| `csc_indices` | int64 | CSC row indices |
| `csc_indptr` | int64 | CSC column pointers (`n_var + 1`) |
| `csc_shape` | int64 | `(n_obs, n_var)` |

Output is always binary (the `.npz`); `--binary` is not required. Values are
promoted to float64 whatever the on-disk dtype. The archive is written
uncompressed (ZIP STORED); total archive size is limited to 4 GiB.

```bash
# Save to a file, then load directly with numpy/scipy
h5ad-inspect data.h5ad export matrix_csr > x_csr.npz
h5ad-inspect data.h5ad export matrix_csc > x_csc.npz
```

```python
import io, subprocess, numpy as np, scipy.sparse as sp

# Capture the stream straight from the process
buf = subprocess.run(
    ["h5ad-inspect", "data.h5ad", "export", "matrix_csr"], capture_output=True
).stdout
z = np.load(io.BytesIO(buf))

csr = sp.csr_matrix(
    (z["csr_data"], z["csr_indices"], z["csr_indptr"]),
    shape=tuple(z["csr_shape"]),
)
# matrix_csc is identical except the members are csc_* and you use sp.csc_matrix.
```

A dense-stored X is emitted as a fully-populated CSR/CSC (every entry kept, so
`nnz == n_obs * n_var`); a sparse-stored X keeps only its stored nonzeros. The
on-disk format and the requested format are independent — e.g.
`export matrix_csc` on a CSR-stored X transposes once to produce CSC.

### 10x CellRanger v3 HDF5 (Seurat)

`export matrix_cellranger_v3_hdf5 <out.h5>` writes the **entire X matrix** as a
10x Genomics CellRanger v3 `.h5` file — the format `Seurat::Read10X_h5()` reads
natively, with no Matrix Market text parsing and no Python/`reticulate`. Unlike
the other matrix exports this one writes to a **file path** you provide (HDF5
cannot stream to stdout), not to the standard output.

10x stores the matrix as **features × barcodes** (genes as rows, cells as
columns) in CSC, i.e. the transpose of AnnData's cells × genes `X`. The export
handles this for you. obs become barcodes (columns), var become features (rows):

| Dataset | dtype | contents |
|---|---|---|
| `matrix/data` | float64 | nonzero values |
| `matrix/indices` | int32 | feature (row) indices, 0-based |
| `matrix/indptr` | int32 | per-barcode column pointers (`n_obs + 1`) |
| `matrix/shape` | int32 | `(n_var, n_obs)` = (features, barcodes) |
| `matrix/barcodes` | string | obs names (column names) |
| `matrix/features/id` | string | var names (mirrors `name`) |
| `matrix/features/name` | string | var names (row names) |
| `matrix/features/feature_type` | string | `"Gene Expression"` for every feature |
| `matrix/features/genome` | string | empty |
| `matrix/features/_all_tag_keys` | string | `["genome"]` |

Index slots are int32 (the `dgCMatrix` ceiling); the export errors out if any
dimension or the nonzero count exceeds 2³¹−1. Works whether X is stored dense,
CSR, or CSC. There is no separate gene-ID column in an AnnData's `var_names`, so
`features/id` mirrors `features/name`.

```bash
h5ad-inspect data.h5ad export matrix_cellranger_v3_hdf5 counts.h5
```

```r
library(Seurat)

# Returns a dgCMatrix: rows = genes (var_names), cols = cells (obs_names)
mat <- Read10X_h5("counts.h5")
dim(mat)            # n_genes x n_cells
head(rownames(mat)) # gene names
head(colnames(mat)) # cell barcodes

# Straight into a Seurat object
obj <- CreateSeuratObject(counts = mat)
```

### Layers (`--layer`)

By default the matrix subcommands read `X`. Pass `--layer <name>` to read from
`layers/<name>` instead — e.g. a `"counts"` layer holding the raw integer counts
behind a normalised `X`. Layers share the obs/var axes (and therefore the cell
and gene names) with `X`, so only the matrix source changes; cell/gene lookups,
sums, and the full-matrix exports all behave exactly as they do for `X`:

```bash
h5ad-inspect data.h5ad export --layer counts row AAACCTGAGAAGGCCT-1
h5ad-inspect data.h5ad export --layer counts column GAPDH
h5ad-inspect data.h5ad export --layer counts obssum
h5ad-inspect data.h5ad export --layer counts matrix_csr > counts_csr.npz
h5ad-inspect data.h5ad export --layer counts matrix_cellranger_v3_hdf5 counts.h5
```

`--layer` applies only to the matrix subcommands (`row`, `column`, `obssum`,
`varsum`, `matrix_csr`, `matrix_csc`, `matrix_cellranger_v3_hdf5`); combining it
with any other subcommand is an error. List the available layer names with
`h5ad-inspect data.h5ad layers`.

> **Note on `.raw`.** AnnData's `.raw` is *not* exposed here. Unlike a layer,
> `.raw` carries its **own** `var` / `var_names` (usually a larger, pre-filtering
> gene set), so it isn't a drop-in alternate matrix on the same axes. If you need
> the raw counts, keep them in a layer (same gene set as `X`) or as a separate
> object.

### Column encoding (`obs_encoding` / `var_encoding`)

Emits a single JSON object describing how a column is stored, so callers can
branch on dtype without reading the whole column:

```bash
# A categorical column includes its categories, in order:
$ h5ad-inspect data.h5ad export obs_encoding cell_type
{"encoding":"categorical","categories":["alpha","beta","gamma"]}

# Boolean and numeric columns carry only the encoding:
$ h5ad-inspect data.h5ad export obs_encoding n_counts
{"encoding":"numeric"}
$ h5ad-inspect data.h5ad export obs_encoding is_control
{"encoding":"bool"}
```

The classification matches the on-disk `encoding-type` attribute: a child group
marked `categorical` is `categorical` (plain string columns are categorical in
recent anndata), a child boolean dataset is `bool`, and everything else
(including nullable-integer groups) is `numeric`. A missing column resolves to
`numeric`.

### Binary float output (`--binary`)

For `row`, `column`, `obssum`, `varsum`, and `obsm` exports, passing `--binary`
writes raw little-endian `float64` bytes to stdout instead of newline-separated
text. This is the fastest path into NumPy — no text parsing overhead.

For `obsm`, the binary stream is row-major: `n_obs * n_components` float64
values, so reshape with `.reshape(n_obs, n_components)` on load.

```bash
h5ad-inspect data.h5ad export --binary row AAACCTGAGAAGGCCT-1 \
  | python3 -c "import numpy as np, sys; print(np.frombuffer(sys.stdin.buffer.read(), dtype='<f8'))"
```

Or save to a file and load later:

```python
import subprocess, numpy as np

# 2-D embedding via binary obsm
result = subprocess.run(
    ["h5ad-inspect", "data.h5ad", "export", "--binary", "obsm", "X_umap"],
    capture_output=True,
)
emb = np.frombuffer(result.stdout, dtype="<f8").reshape(-1, 2)
```

The output is always `float64` regardless of the on-disk dtype (float32, int32, etc.).

---

## Shell completions (Fish)

```bash
# Manual install
cp completions/h5ad-inspect.fish ~/.config/fish/completions/
```

Completions dynamically query the open file for obs/var column names and index
values, so tab-completion works for actual cell and gene IDs.
