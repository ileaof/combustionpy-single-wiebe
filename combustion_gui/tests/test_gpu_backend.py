# -*- coding: utf-8 -*-
"""Testes do modo acelerado (RK4 em lote NumPy / CUDA) da GUI Single Wiebe.

Os testes de CUDA são pulados sem CuPy/GPU. Executar de combustion_gui/:
    python -m pytest tests/ -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

import data_processing as dp
import gpu_backend as gpu
import optimization as opt
import single_wiebe as sw

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "P_exp-Carga-3_45%.txt"
VALID = (17.0, 0.504, math.radians(-6.54), math.radians(62.0))
requer_gpu = pytest.mark.skipif(not gpu.cuda_available(),
                                reason="CuPy/GPU CUDA indisponível")


@pytest.fixture(scope="module")
def data():
    theta, P, _ = dp.load_file_full(
        SAMPLE, angle_col=0, pressure_col=1, angle_unit="radianos",
        pressure_unit="bar", theta_min=-2.0, theta_max=2.0,
    )
    return theta, P


@pytest.fixture(scope="module")
def populacao():
    rng = np.random.default_rng(0)
    X = sw.LOWER + rng.random((40, 4)) * (sw.UPPER - sw.LOWER)
    X[0] = VALID
    return X


# ---------------------------------------------------------------------------
# RK4 em lote (CPU) vs referência solve_ivp
# ---------------------------------------------------------------------------
def test_lote_cpu_validacao_notebook(data):
    """Caso de validação do notebook: RK4 em lote ≈ solve_ivp (409.3856)."""
    theta, P = data
    f = gpu.make_batch_objective(theta, P, backend="cpu")
    erro = f(np.array([VALID]))[0]
    assert erro == pytest.approx(409.385649, abs=0.1)


def test_lote_cpu_vs_solve_ivp(data, populacao):
    """Onde a referência converge, |Δerro| < 5 kPa; falha do DOP853 só
    ocorre na fronteira (m pequeno), nunca no sentido inverso."""
    theta, P = data
    f = gpu.make_batch_objective(theta, P, backend="cpu")(populacao)
    ref = np.array([sw.simulate_model(*x, theta, P)[0] for x in populacao])
    assert np.isfinite(f).all()
    ok = ref < sw.PENALTY
    assert ok.sum() >= 30
    assert (f[ok] < sw.PENALTY).all()
    assert np.max(np.abs(f[ok] - ref[ok])) < 5.0


def test_lote_candidatos_invalidos_penalidade(data):
    theta, P = data
    X = np.array([VALID, VALID, VALID], dtype=float)
    X[0, 0] = 1.0            # Rc <= 1
    X[1, 3] = 0.0            # delta_theta <= 0
    f = gpu.make_batch_objective(theta, P, backend="cpu")(X)
    assert f[0] == sw.PENALTY and f[1] == sw.PENALTY and f[2] < sw.PENALTY


def test_calibracao_cpu_lote_valida_com_referencia(data):
    theta, P = data
    r = opt.run_calibration(
        theta, P, ["m", "theta0"], opt.DEFAULT_BOUNDS,
        {"Rc": 17.0, "m": 0.5, "theta0": 0.0, "delta_theta": math.radians(50)},
        sw.EngineConfig(), method="PSO", n_particulas=8, maxiter=10, seed=42,
        backend="cpu-lote")
    v = r["validacao"]
    assert v["backend"] == "cpu-lote"
    # erro reportado = referência (solve_ivp) no melhor candidato
    ref = sw.simulate_model(*r["params"], theta, P,
                            method=v["integrador"])[0]
    assert r["erro"] == pytest.approx(ref, rel=1e-12)
    assert v["diferenca"] < 5.0


def test_serial_nao_muda(data):
    """backend serial = caminho original (sem validação extra)."""
    theta, P = data
    kw = dict(theta_exp=theta, pressure_exp=P, selected=["m"],
              bounds=opt.DEFAULT_BOUNDS,
              x_init={"Rc": 17.0, "m": 0.5, "theta0": 0.0,
                      "delta_theta": math.radians(50)},
              cfg=sw.EngineConfig(), method="PSO", n_particulas=4,
              maxiter=3, seed=1)
    r = opt.run_calibration(**kw)
    assert r["validacao"] is None


def test_backend_invalido(data):
    theta, P = data
    with pytest.raises(opt.OptimizationError):
        opt.run_calibration(theta, P, ["m"], opt.DEFAULT_BOUNDS,
                            {"Rc": 17.0, "m": 0.5, "theta0": 0.0,
                             "delta_theta": 1.0},
                            sw.EngineConfig(), backend="tpu")


def test_de_em_lote(data):
    theta, P = data
    r = opt.run_calibration(
        theta, P, ["m"], opt.DEFAULT_BOUNDS,
        {"Rc": 17.0, "m": 0.5, "theta0": 0.0, "delta_theta": math.radians(50)},
        sw.EngineConfig(), method="DE", n_particulas=10, maxiter=5, seed=3,
        backend="cpu-lote")
    lo, hi = opt.DEFAULT_BOUNDS["m"]
    assert lo <= r["params"][1] <= hi
    assert r["erro"] < sw.PENALTY


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
@requer_gpu
def test_cuda_igual_numpy(data, populacao):
    theta, P = data
    a = gpu.batch_rk4_numpy(theta, P[0], populacao)
    g = gpu.batch_rk4_cuda(theta, P[0], populacao)
    assert np.array_equal(np.isnan(a).any(axis=1), np.isnan(g).any(axis=1))
    ok = np.isfinite(a)
    np.testing.assert_allclose(g[ok], a[ok], rtol=1e-12, atol=0.0)


@requer_gpu
def test_cuda_objetivo_igual_cpu(data, populacao):
    theta, P = data
    f_cpu = gpu.make_batch_objective(theta, P, backend="cpu")(populacao)
    f_gpu = gpu.make_batch_objective(theta, P, backend="cuda")(populacao)
    np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-10)
    f32 = gpu.make_batch_objective(theta, P, backend="cuda",
                                   precision="float32")(populacao)
    assert np.max(np.abs(f32 - f_cpu)) < 0.1


@requer_gpu
def test_cuda_falhas_iguais_cpu(data):
    """Candidatos não físicos falham nos mesmos pontos em CPU e GPU."""
    theta, P = data
    X = np.array([VALID] * 5, dtype=float)
    X[0, 1] = -3.0                      # m < 0 (integra: exp -> 0)
    X[1, 0] = 1.0                       # Rc <= 1
    X[2, 3] = -0.1                      # delta_theta < 0
    X[3, 0] = 0.5                       # Rc < 1 -> volume negativo
    f_cpu = gpu.make_batch_objective(theta, P, backend="cpu")(X)
    f_gpu = gpu.make_batch_objective(theta, P, backend="cuda")(X)
    assert np.array_equal(f_cpu >= sw.PENALTY, f_gpu >= sw.PENALTY)
    assert (f_gpu[1:4] >= sw.PENALTY).all()
    assert f_gpu[0] < sw.PENALTY and f_gpu[4] < sw.PENALTY
    ok = f_cpu < sw.PENALTY
    np.testing.assert_allclose(f_gpu[ok], f_cpu[ok], rtol=1e-10)


@requer_gpu
def test_calibracao_cuda(data):
    theta, P = data
    x0 = {"Rc": 17.0, "m": 0.5, "theta0": 0.0, "delta_theta": math.radians(50)}
    base = dict(method="PSO", n_particulas=10, maxiter=15, seed=42)
    r_cpu = opt.run_calibration(theta, P, list(opt.PARAM_NAMES),
                                opt.DEFAULT_BOUNDS, x0, sw.EngineConfig(),
                                backend="cpu-lote", **base)
    r_gpu = opt.run_calibration(theta, P, list(opt.PARAM_NAMES),
                                opt.DEFAULT_BOUNDS, x0, sw.EngineConfig(),
                                backend="cuda", precision="float64", **base)
    # mesmo integrador em lote -> mesma trajetória do PSO
    np.testing.assert_allclose(r_gpu["params"], r_cpu["params"], rtol=1e-9)
    assert r_gpu["erro"] == pytest.approx(r_cpu["erro"], rel=1e-9)
    assert r_gpu["validacao"]["diferenca"] < 5.0
