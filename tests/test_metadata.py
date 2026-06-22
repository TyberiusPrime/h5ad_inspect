"""Tests for shape, index (export + inspect), and column listings."""

from __future__ import annotations

import pytest

from conftest import run, text_lines
import reference as ref


def test_shape(h5ad_inspect, files):
    adata, path = files["dense"]
    lines = text_lines(run(h5ad_inspect, path, "shape"))
    assert lines == [
        f"n_obs\t{adata.shape[0]}",
        f"n_var\t{adata.shape[1]}",
    ]


@pytest.mark.parametrize("axis,name", [(0, "obs_index"), (1, "var_index")])
def test_export_index_original_order(h5ad_inspect, files, axis, name):
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, "export", name))
    assert got == ref.expected_index(adata, axis)


@pytest.mark.parametrize("axis,name", [(0, "obs_index"), (1, "var_index")])
def test_inspect_index_is_sorted(h5ad_inspect, files, axis, name):
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, name))
    assert got == sorted(ref.expected_index(adata, axis))


@pytest.mark.parametrize("group", ["obs", "var"])
def test_inspect_lists_columns(h5ad_inspect, files, group):
    """`h5ad-inspect f <group>` must list exactly the real columns, sorted."""
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, group))
    expected = sorted(getattr(adata, group).columns)
    assert got == expected


def test_inspect_obsm_keys(h5ad_inspect, files):
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, "obsm"))
    assert got == sorted(adata.obsm.keys())


def test_inspect_uns_keys(h5ad_inspect, files):
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, "uns"))
    assert got == sorted(adata.uns.keys())
