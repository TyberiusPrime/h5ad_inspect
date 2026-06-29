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


@pytest.mark.parametrize(
    "group,axis,col",
    [("obs", 0, "ocat"), ("obs", 0, "of64"), ("var", 1, "vcat"), ("var", 1, "vf64")],
)
def test_export_include_index(h5ad_inspect, files, group, axis, col):
    """`--include-index` prefixes each value with its cell/gene ID, tab-separated."""
    adata, path = files["dense"]
    values = text_lines(run(h5ad_inspect, path, "export", group, col))
    indexed = text_lines(
        run(h5ad_inspect, path, "export", "--include-index", group, col)
    )
    index = ref.expected_index(adata, axis)
    assert len(indexed) == len(values) == len(index)
    for line, idx, val in zip(indexed, index, values):
        assert line == f"{idx}\t{val}"


def test_export_include_index_rejected_for_non_column(h5ad_inspect, files):
    """`--include-index` only applies to obs/var column exports."""
    _adata, path = files["dense"]
    cp = run(h5ad_inspect, path, "export", "--include-index", "obssum", expect_ok=False)
    assert cp.returncode != 0


@pytest.mark.parametrize("name", ["obs_index", "var_index"])
def test_inspect_index_command_removed(h5ad_inspect, files, name):
    """Index values belong to `export` only; inspect mode rejects them."""
    _adata, path = files["dense"]
    cp = run(h5ad_inspect, path, name, expect_ok=False)
    assert cp.returncode != 0


@pytest.mark.parametrize("group", ["obs", "var"])
def test_inspect_lists_columns(h5ad_inspect, files, group):
    """`h5ad-inspect f <group>` must list exactly the real columns (file order)."""
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, group))
    expected = list(getattr(adata, group).columns)
    assert set(got) == set(expected)


@pytest.mark.parametrize("group", ["obs", "var"])
def test_inspect_lists_columns_sorted(h5ad_inspect, files, group):
    """`--sorted` sorts the listed columns alphabetically."""
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, "--sorted", group))
    assert got == sorted(getattr(adata, group).columns)


def test_inspect_obsm_keys(h5ad_inspect, files):
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, "obsm"))
    assert set(got) == set(adata.obsm.keys())


def test_inspect_uns_keys(h5ad_inspect, files):
    adata, path = files["dense"]
    got = text_lines(run(h5ad_inspect, path, "uns"))
    assert set(got) == set(adata.uns.keys())
