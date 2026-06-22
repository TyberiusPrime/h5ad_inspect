"""Tests for X-matrix and obsm exports, across dense, CSR and CSC X layouts."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import run, to_float_array, to_float_matrix, to_binary_floats
import reference as ref

# Sums accumulate, so allow a touch more tolerance for float32 summation
# differences between Rust (f64) and numpy (float32 in-place).
SUM_RTOL = 1e-5
SUM_ATOL = 1e-6
ELEM_RTOL = 1e-6
ELEM_ATOL = 1e-9

VARIANTS = ["dense", "csr", "csc"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_obssum(h5ad_inspect, files, variant):
    adata, path = files[variant]
    got = to_float_array(run(h5ad_inspect, path, "export", "obssum"))
    expected = ref.expected_obssum(adata)
    assert np.allclose(got, expected, rtol=SUM_RTOL, atol=SUM_ATOL), (variant, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_varsum(h5ad_inspect, files, variant):
    adata, path = files[variant]
    got = to_float_array(run(h5ad_inspect, path, "export", "varsum"))
    expected = ref.expected_varsum(adata)
    assert np.allclose(got, expected, rtol=SUM_RTOL, atol=SUM_ATOL), (variant, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_row(h5ad_inspect, files, variant):
    adata, path = files[variant]
    obs_name = str(adata.obs_names[3])
    got = to_float_array(run(h5ad_inspect, path, "export", "row", obs_name))
    expected = ref.expected_x_row(adata, obs_name)
    assert np.allclose(got, expected, rtol=ELEM_RTOL, atol=ELEM_ATOL), (variant, obs_name, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_row_unicode(h5ad_inspect, files, variant):
    """A row lookup by a unicode obs index value must resolve correctly."""
    adata, path = files[variant]
    obs_name = "célL1"
    got = to_float_array(run(h5ad_inspect, path, "export", "row", obs_name))
    expected = ref.expected_x_row(adata, obs_name)
    assert np.allclose(got, expected, rtol=ELEM_RTOL, atol=ELEM_ATOL), (variant, obs_name, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_column(h5ad_inspect, files, variant):
    adata, path = files[variant]
    var_name = str(adata.var_names[5])
    got = to_float_array(run(h5ad_inspect, path, "export", "column", var_name))
    expected = ref.expected_x_col(adata, var_name)
    assert np.allclose(got, expected, rtol=ELEM_RTOL, atol=ELEM_ATOL), (variant, var_name, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_column_unicode(h5ad_inspect, files, variant):
    adata, path = files[variant]
    var_name = "Gène2"
    got = to_float_array(run(h5ad_inspect, path, "export", "column", var_name))
    expected = ref.expected_x_col(adata, var_name)
    assert np.allclose(got, expected, rtol=ELEM_RTOL, atol=ELEM_ATOL), (variant, var_name, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("key", ["X_pca", "X_umap"])
def test_obsm(h5ad_inspect, files, variant, key):
    adata, path = files[variant]
    got = to_float_matrix(run(h5ad_inspect, path, "export", "obsm", key))
    expected = ref.expected_obsm(adata, key)
    assert got.shape == expected.shape, (variant, key, got.shape, expected.shape)
    assert np.allclose(got, expected, rtol=ELEM_RTOL, atol=ELEM_ATOL), (variant, key, got, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_obsm_binary(h5ad_inspect, files, variant):
    adata, path = files[variant]
    got = to_binary_floats(run(h5ad_inspect, path, "export", "--binary", "obsm", "X_umap"))
    expected = ref.expected_obsm(adata, "X_umap")
    assert got.reshape(expected.shape).shape == expected.shape
    assert np.allclose(got, expected.ravel(), rtol=ELEM_RTOL, atol=ELEM_ATOL), (variant, got, expected.ravel())


@pytest.mark.parametrize("variant", VARIANTS)
def test_obssum_binary_matches_text(h5ad_inspect, files, variant):
    _, path = files[variant]
    binary = to_binary_floats(run(h5ad_inspect, path, "export", "--binary", "obssum"))
    text = to_float_array(run(h5ad_inspect, path, "export", "obssum"))
    assert np.allclose(binary, text, equal_nan=True), (variant, binary, text)
