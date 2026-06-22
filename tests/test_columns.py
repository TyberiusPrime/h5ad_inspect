"""Tests for `export obs|var <col>` and `export obs_categories|var_categories <col>`.

These cover every column dtype anndata can write: floats, signed/unsigned
ints, bool, plain strings, ordered/unordered categoricals, nullable integers,
nullable strings, and floats with NaN. Column storage is independent of the X
layout, so all of these run against the dense variant.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import run, text_lines
import reference as ref

NUMERIC_RTOL = 1e-6
NUMERIC_ATOL = 1e-9


def _column_cases(adata, group):
    return [(group, c) for c in getattr(adata, group).columns]


@pytest.fixture(scope="module")
def dense(files):
    return files["dense"]


@pytest.mark.parametrize("group,col", [
    *(("obs", c) for c in ["of64", "of32", "oi32", "oi8", "ou8", "obool",
                           "ostr", "ocat", "ocatord", "onint", "ofnan"]),
    *(("var", c) for c in ["vf64", "vf32", "vi64", "vi16", "vi8", "vu8", "vbool",
                           "vstr", "vcat", "vnint", "vnstr", "vfnan"]),
])
def test_export_column(h5ad_inspect, dense, group, col):
    adata, path = dense
    cp = run(h5ad_inspect, path, "export", group, col)
    expected = ref.expected_columns(adata, group, col)

    if isinstance(expected, np.ndarray):
        got = np.array(
            [np.nan if ln in ("NA", "NaN", "nan") else float(ln) for ln in text_lines(cp)],
            dtype=float,
        )
        assert got.shape == expected.shape, (col, got.shape, expected.shape)
        assert np.allclose(got, expected, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL,
                           equal_nan=True), (col, got, expected)
    else:
        got = text_lines(cp)
        assert got == expected, (col, got, expected)


@pytest.mark.parametrize("group,col", [
    ("obs", "ocat"),
    ("obs", "ocatord"),
    ("var", "vcat"),
    ("var", "vnstr"),  # nullable-string categorical → categories may be a group
])
def test_export_categories(h5ad_inspect, dense, group, col):
    adata, path = dense
    got = text_lines(run(h5ad_inspect, path, "export", f"{group}_categories", col))
    assert got == ref.expected_categories(adata, group, col), (col, got)


def test_export_nonexistent_column_errors(h5ad_inspect, dense):
    """A genuinely missing column must fail (non-zero exit), not silently emit."""
    _, path = dense
    cp = run(h5ad_inspect, path, "export", "var", "this_column_does_not_exist",
             expect_ok=False)
    assert cp.returncode != 0


@pytest.mark.parametrize("group,col", [
    *(("obs", c) for c in ["of64", "of32", "oi32", "oi8", "ou8", "obool",
                           "ostr", "ocat", "ocatord", "onint", "ofnan"]),
    *(("var", c) for c in ["vf64", "vf32", "vi64", "vi16", "vi8", "vu8", "vbool",
                           "vstr", "vcat", "vnint", "vnstr", "vfnan"]),
])
def test_export_encoding(h5ad_inspect, dense, group, col):
    """obs_encoding/var_encoding emit JSON matching the h5py reference."""
    adata, path = dense
    got = json.loads(
        run(h5ad_inspect, path, "export", f"{group}_encoding", col).stdout.decode()
    )
    expected = ref.expected_encoding(path, group, col)
    assert got == expected, (group, col, got, expected)


def test_export_encoding_missing_column_is_numeric(h5ad_inspect, dense):
    """A missing column resolves to numeric (mirrors the h5py reference)."""
    _, path = dense
    got = json.loads(
        run(h5ad_inspect, path, "export", "obs_encoding",
            "this_column_does_not_exist").stdout.decode()
    )
    assert got == {"encoding": "numeric"}


@pytest.mark.parametrize("special", ['"', "\\", "\n", "\t", "café"])
def test_export_encoding_escapes_categories(h5ad_inspect, files, special):
    """Category strings must be valid JSON regardless of special characters."""
    adata, base_path = files["dense"]
    n_obs = adata.shape[0]
    # Rewrite with a categorical whose category contains special characters.
    path = base_path.with_name("special_cat.h5ad")
    a = adata.copy()
    vals = [special if i % 2 == 0 else "plain" for i in range(n_obs)]
    a.obs["sp"] = pd.Categorical(vals, categories=[special, "plain"])
    a.write_h5ad(path)
    got = json.loads(run(h5ad_inspect, path, "export", "obs_encoding", "sp").stdout.decode())
    assert got["encoding"] == "categorical"
    assert got["categories"] == [special, "plain"]
