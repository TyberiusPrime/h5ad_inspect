"""Shared fixtures: the h5ad-inspect binary and h5ad test files generated with anndata.

Every fixture h5ad is generated on the fly from a single AnnData object so the
suite is hermetic (no multi-GB files checked in) and exercises every column
dtype / X layout we support. Outputs of `h5ad-inspect` are compared against the
reference values read straight back out of the same AnnData object.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ad.settings.allow_write_nullable_strings = True

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = REPO_ROOT / "h5ad_inspect"


def build_adata() -> ad.AnnData:
    """A single AnnData object covering every storage shape we care about."""
    n_obs, n_var = 6, 7
    rng = np.random.default_rng(0)

    obs_names = [f"c{i}" for i in range(n_obs)]
    obs_names[1] = "célL1"  # unicode obs index
    var_names = [f"G{i}" for i in range(n_var)]
    var_names[2] = "Gène2"  # unicode var index

    X = rng.standard_normal((n_obs, n_var)).astype("float32")

    obs = pd.DataFrame(index=pd.Index(obs_names, name="obs_id"))
    obs["of64"] = rng.standard_normal(n_obs)
    obs["of32"] = rng.standard_normal(n_obs).astype("float32")
    obs["oi32"] = rng.integers(-50, 50, n_obs).astype("int32")
    obs["oi8"] = rng.integers(-100, 100, n_obs).astype("int8")
    obs["ou8"] = rng.integers(0, 200, n_obs).astype("uint8")
    obs["obool"] = rng.random(n_obs) > 0.5
    obs["ostr"] = [f"s{i % 3}" for i in range(n_obs)]
    obs["ocat"] = pd.Categorical(rng.choice(["alpha", "beta", "gamma"], n_obs))
    obs["ocatord"] = pd.Categorical(
        rng.choice(["lo", "mid", "hi"], n_obs),
        categories=["lo", "mid", "hi"],
        ordered=True,
    )
    obs["onint"] = pd.array([1, 2, None, 4, 5, None], dtype="Int32")
    obs["ofnan"] = [1.0, np.nan, 3.0, 4.0, np.nan, 6.0]

    var = pd.DataFrame(index=pd.Index(var_names, name="var_id"))
    var["vf64"] = rng.standard_normal(n_var)
    var["vf32"] = rng.standard_normal(n_var).astype("float32")
    var["vi64"] = rng.integers(0, 1000, n_var).astype("int64")
    var["vi16"] = rng.integers(-1000, 1000, n_var).astype("int16")
    var["vi8"] = rng.integers(-100, 100, n_var).astype("int8")
    var["vu8"] = rng.integers(0, 255, n_var).astype("uint8")
    var["vbool"] = rng.random(n_var) > 0.5
    var["vstr"] = [f"g{i % 4}" for i in range(n_var)]
    var["vcat"] = pd.Categorical(rng.choice(["x", "y", "z"], n_var))
    var["vnint"] = pd.array([10, None, 30, 40, None, 60, 70], dtype="Int32")
    var["vnstr"] = pd.array(["p", "q", None, "r", "p", None, "s"], dtype="string")
    var["vfnan"] = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0]

    a = ad.AnnData(X=X, obs=obs, var=var)
    a.obsm["X_pca"] = rng.standard_normal((n_obs, 3)).astype("float32")
    a.obsm["X_umap"] = rng.standard_normal((n_obs, 2))
    # A layer distinct from X (integer-valued counts) so --layer exports can be
    # told apart from the default X export. Shares obs/var axes with X.
    a.layers["counts"] = rng.integers(0, 10, (n_obs, n_var)).astype("float32")
    return a


@pytest.fixture(scope="session")
def h5ad_inspect() -> str:
    """Path to the h5ad-inspect binary.

    Honours $H5AD_INSPECT_BIN (set by the Nix check); otherwise builds the
    release binary once via cargo.
    """
    env = os.environ.get("H5AD_INSPECT_BIN")
    if env and Path(env).exists():
        return env

    target = RUST_ROOT / "target"
    subprocess.run(
        ["cargo", "build", "--release", "--target-dir", str(target)],
        cwd=RUST_ROOT,
        check=True,
    )
    binary = target / "release" / "h5ad-inspect"
    assert binary.exists(), binary
    return str(binary)


@pytest.fixture(scope="session")
def files(tmp_path_factory):
    """Write the AnnData in three X layouts: dense, CSR, CSC.

    Returns ``{variant: (AnnData, h5ad_path)}``. Column/index storage is
    identical across variants, so column tests use the dense variant while
    X-dependent tests are parametrised over all three.
    """
    out_dir = tmp_path_factory.mktemp("h5ad")
    base = build_adata()
    dense_X = np.asarray(base.X)
    dense_counts = np.asarray(base.layers["counts"])

    variants = {}
    for variant in ("dense", "csr", "csc"):
        a = base.copy()
        if variant == "dense":
            a.X = dense_X.copy()
            a.layers["counts"] = dense_counts.copy()
        elif variant == "csr":
            a.X = sp.csr_matrix(dense_X)
            a.layers["counts"] = sp.csr_matrix(dense_counts)
        elif variant == "csc":
            a.X = sp.csc_matrix(dense_X)
            a.layers["counts"] = sp.csc_matrix(dense_counts)
        path = out_dir / f"{variant}.h5ad"
        a.write_h5ad(path)
        variants[variant] = (a, path)
    return variants


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for invoking the tool and decoding its output.
# ──────────────────────────────────────────────────────────────────────────────


def run(h5ad_inspect: str, h5ad: Path, *rest: str, expect_ok: bool = True):
    """Run `h5ad-inspect <h5ad> <rest...>` and return the CompletedProcess."""
    cp = subprocess.run(
        [h5ad_inspect, str(h5ad), *rest],
        capture_output=True,
    )
    if expect_ok:
        assert cp.returncode == 0, (
            f"command failed (rc={cp.returncode}): "
            f"h5ad-inspect {h5ad.name} {' '.join(rest)}\n{cp.stderr.decode()}"
        )
    return cp


def text_lines(cp) -> list[str]:
    return cp.stdout.decode().splitlines()


def to_float_array(cp) -> np.ndarray:
    """Parse newline-separated numeric text, mapping NA/NaN → nan."""
    return np.array(
        [np.nan if ln in ("NA", "NaN", "nan") else float(ln) for ln in text_lines(cp)],
        dtype=float,
    )


def to_float_matrix(cp) -> np.ndarray:
    """Parse tab-separated numeric rows, mapping NA/NaN → nan."""
    return np.array(
        [
            [np.nan if x in ("NA", "NaN", "nan") else float(x) for x in ln.split("\t")]
            for ln in text_lines(cp)
        ],
        dtype=float,
    )


def to_binary_floats(cp) -> np.ndarray:
    return np.frombuffer(cp.stdout, dtype="<f8")
