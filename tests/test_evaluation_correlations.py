import numpy as np
import pytest

from vgep.evaluation.correlations import per_gene_pearson_r


def _rng():
    return np.random.default_rng(42)


def test_identity_gives_ones():
    rng = _rng()
    ref = rng.normal(size=(5, 12)).astype(np.float32)
    r = per_gene_pearson_r(ref, ref)
    assert r.shape == (5,)
    assert r.dtype == np.float32
    np.testing.assert_allclose(r, 1.0, atol=1e-5)


def test_anticorrelation_gives_minus_one():
    rng = _rng()
    ref = rng.normal(size=(3, 20)).astype(np.float32)
    r = per_gene_pearson_r(-ref, ref)
    np.testing.assert_allclose(r, -1.0, atol=1e-5)


def test_shape_output_matches_gene_axis_size():
    rng = _rng()
    pred = rng.normal(size=(7, 30)).astype(np.float32)
    ref = rng.normal(size=(7, 30)).astype(np.float32)
    r = per_gene_pearson_r(pred, ref)
    assert r.shape == (7,)


def test_gene_axis_trailing():
    # Shape (n_bins, G) with gene_axis=1
    rng = _rng()
    data = rng.normal(size=(30, 5)).astype(np.float32)
    r = per_gene_pearson_r(data, data, gene_axis=1)
    assert r.shape == (5,)
    np.testing.assert_allclose(r, 1.0, atol=1e-5)


def test_all_nan_gene_returns_nan():
    rng = _rng()
    pred = rng.normal(size=(3, 20)).astype(np.float32)
    ref = rng.normal(size=(3, 20)).astype(np.float32)
    pred[1, :] = np.nan
    r = per_gene_pearson_r(pred, ref)
    assert np.isnan(r[1])
    assert np.isfinite(r[0]) and np.isfinite(r[2])


def test_constant_gene_returns_nan_no_warning():
    rng = _rng()
    pred = rng.normal(size=(3, 20)).astype(np.float32)
    ref = rng.normal(size=(3, 20)).astype(np.float32)
    pred[1, :] = 5.0  # zero variance
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        r = per_gene_pearson_r(pred, ref)
    assert np.isnan(r[1])
    assert np.isfinite(r[0]) and np.isfinite(r[2])


def test_partial_nan_masking():
    # Gene 0: first half of ref is nan; per-gene mask should drop those bins, r on remaining should be perfect
    rng = _rng()
    ref = rng.normal(size=(2, 40)).astype(np.float32)
    pred = ref.copy()
    ref[0, :20] = np.nan
    r = per_gene_pearson_r(pred, ref, mask_nan=True)
    np.testing.assert_allclose(r, 1.0, atol=1e-5)
