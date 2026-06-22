"""Build the expected h5ad-inspect output from an AnnData object (the reference).

These map the in-memory values to the exact strings/floats the tool emits, so a
mismatch is a real bug and not a formatting quirk.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp


def is_sparse(X) -> bool:
    return sp.issparse(X)


def to_dense_1d(X) -> np.ndarray:
    if is_sparse(X):
        return np.asarray(X.todense()).ravel()
    return np.asarray(X).ravel()


def na_str(v) -> str:
    if v is None:
        return "NA"
    try:
        if isinstance(v, float) and math.isnan(v):
            return "NA"
    except (TypeError, ValueError):
        pass
    if v is pd.NA:
        return "NA"
    return str(v)


def expected_index(adata, axis: int) -> list[str]:
    names = adata.obs_names if axis == 0 else adata.var_names
    return [str(n) for n in names]


def expected_columns(adata, group: str, col: str):
    """Return either a numeric ndarray or a list[str] to compare against.

    - bool columns → ["true"/"false"] (the tool's lowercase spelling)
    - numeric (incl. nullable) → float ndarray (NA/NaN → nan)
    - string / categorical → list[str] with NA → "NA"
    """
    series = getattr(adata, group)[col]
    dtype = series.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        vals = series.astype("object").to_numpy()
        return [na_str(v) for v in vals]

    if dtype == bool:
        return ["true" if bool(v) else "false" for v in series.to_numpy()]

    if pd.api.types.is_numeric_dtype(dtype):
        return series.to_numpy(dtype=float)

    # object / nullable-string
    return [na_str(v) for v in series.to_numpy()]


def expected_categories(adata, group: str, col: str) -> list[str]:
    series = getattr(adata, group)[col]
    cats = series.cat.categories
    return [na_str(v) for v in cats]


def expected_x_row(adata, obs_name: str) -> np.ndarray:
    return to_dense_1d(adata[obs_name, :].X).astype(float)


def expected_x_col(adata, var_name: str) -> np.ndarray:
    return to_dense_1d(adata[:, var_name].X).astype(float)


def expected_obssum(adata) -> np.ndarray:
    return np.asarray(adata.X.sum(axis=1)).ravel().astype(float)


def expected_varsum(adata) -> np.ndarray:
    return np.asarray(adata.X.sum(axis=0)).ravel().astype(float)


def expected_obsm(adata, key: str) -> np.ndarray:
    return np.asarray(adata.obsm[key]).astype(float)


def expected_encoding(path, group: str, col: str) -> dict:
    """Reference h5ad encoding for a column, read straight from the file.

    Mirrors the h5py-based ``_col_encoding``: ``categorical`` / ``bool`` /
    ``numeric``. Categorical columns also carry their in-order ``categories``
    (decoded the same way the tool decodes them, with masked entries → "NA").
    """
    import h5py

    def _decode(v) -> str:
        if isinstance(v, bytes):
            return v.decode()
        return str(v)

    with h5py.File(path, "r") as f:
        grp = f.get(group)
        enc = "numeric"
        if isinstance(grp, h5py.Group) and col in grp:
            item = grp[col]
            if isinstance(item, h5py.Group):
                e = item.attrs.get("encoding-type", b"")
                if isinstance(e, bytes):
                    e = e.decode()
                if e == "categorical":
                    enc = "categorical"
            elif isinstance(item, h5py.Dataset) and item.dtype.kind == "b":
                enc = "bool"

        out: dict[str, object] = {"encoding": enc}
        if enc == "categorical":
            cat = f.get(f"{group}/{col}/categories")
            cats: list[str] = []
            if isinstance(cat, h5py.Dataset):
                cats = [_decode(v) for v in np.asarray(cat[()]).ravel()]
            elif isinstance(cat, h5py.Group):
                vals_ds = cat.get("values")
                vals = (
                    np.asarray(vals_ds[()]).ravel()
                    if isinstance(vals_ds, h5py.Dataset)
                    else np.asarray([])
                )
                mask_ds = cat.get("mask")
                mask = (
                    np.asarray(mask_ds[()]).ravel()
                    if isinstance(mask_ds, h5py.Dataset)
                    else None
                )
                for i, v in enumerate(vals):
                    if mask is not None and mask[i]:
                        cats.append("NA")
                    else:
                        cats.append(_decode(v))
            out["categories"] = cats
        return out
