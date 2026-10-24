"""Velocity model shapes, normalization, and the on-device export wrapper."""

import numpy as np
import pytest
import torch

from src.models.velocity_model import VelocityCNNDeep, apply_normalization, compute_normalization_stats


def test_normalization_uses_given_stats():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, size=(50, 20, 3)).astype(np.float32)
    stats = compute_normalization_stats(X)
    Xn = apply_normalization(X, stats)
    assert np.allclose(Xn.reshape(-1, 3).mean(0), 0.0, atol=1e-4)
    assert np.allclose(Xn.reshape(-1, 3).std(0), 1.0, atol=1e-3)
    assert Xn.dtype == np.float32


def test_deep_cnn_output_shape():
    m = VelocityCNNDeep(n_features=8, channels=(8, 16)).eval()
    assert m(torch.zeros(4, 8, 200)).shape == (4, 1)


def _small_model_and_stats():
    torch.manual_seed(0)
    m = VelocityCNNDeep(n_features=8, channels=(8, 16)).eval()
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(64, 200, 8)).astype(np.float32) * 3 + 1
    return m, compute_normalization_stats(X), X


def test_deployable_wrapper_matches_manual_pipeline():
    from src.models.compress_model import DeployableVelocityModel, predict_raw

    m, stats, X = _small_model_and_stats()
    ref = m(torch.from_numpy(apply_normalization(X, stats).transpose(0, 2, 1).copy())).squeeze(-1).detach().numpy()
    got = predict_raw(DeployableVelocityModel(m, stats).eval(), X)
    assert np.allclose(ref, got, atol=1e-5)


def test_lite_export_round_trip(tmp_path):
    from src.models.compress_model import DeployableVelocityModel, export_lite, load_lite, predict_raw

    m, stats, X = _small_model_and_stats()
    d = DeployableVelocityModel(m, stats).eval()
    lite = load_lite(export_lite(d, tmp_path / "m.ptl", 200, 8))
    assert np.allclose(predict_raw(d, X), predict_raw(lite, X), atol=1e-4)


@pytest.mark.skipif(not ({"x86", "fbgemm", "qnnpack"} & set(torch.backends.quantized.supported_engines)),
                    reason="no quantized engine")
def test_int8_quantization_stays_close():
    from src.models.compress_model import DeployableVelocityModel, predict_raw, quantize_int8

    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else "x86"
    m, stats, X = _small_model_and_stats()
    q = DeployableVelocityModel(quantize_int8(m, X, stats, 200, 8), stats).eval()
    ref = predict_raw(DeployableVelocityModel(m, stats).eval(), X)
    assert np.mean(np.abs(predict_raw(q, X) - ref)) < 0.1 * (np.std(ref) + 1e-3) + 0.05
