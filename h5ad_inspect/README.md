# h5ad-inspect

A fast command-line tool for inspecting and exporting data from [AnnData](https://anndata.readthedocs.io/) `.h5ad` files.

## Installation

```
cargo install --path .
```

## Usage

```
h5ad-inspect <filename> <section>
h5ad-inspect <filename> export [--binary] <subcommand> [<name>]
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
| `obsm` | key | a 2-D obsm embedding (e.g. `X_pca`, `X_umap`) in obs order; one tab-separated line per obs, `n_components` values per line |
| `obs_categories` | column name | order of the categories  of this column |
| `var_categories` | column name | order of the categories  of this column |
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
