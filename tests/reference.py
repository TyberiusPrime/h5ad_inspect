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
