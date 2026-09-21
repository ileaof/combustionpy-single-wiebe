# -*- coding: utf-8 -*-
"""
gpu_backend.py
===============================================================================
Avaliação EM LOTE da função objetivo do Single Wiebe (modo acelerado).

Integra N candidatos [Rc, m, theta0, delta_theta] de uma vez com RK4
clássico de passo fixo (``substeps`` sub-passos por intervalo entre ângulos
experimentais). As equações são EXATAMENTE as de ``single_wiebe._make_rhs``
(geometria, Wiebe sem truncamento, Hohenberg); muda apenas o integrador
(solve_ivp/DOP853 adaptativo -> RK4 de passo fixo).

* ``batch_rk4_numpy`` — referência do integrador em lote na CPU (NumPy);
* ``batch_rk4_cuda``  — o mesmo laço na GPU NVIDIA (CuPy, opcional):
  um thread CUDA por candidato;
* ``make_batch_objective`` — erro sqrt(SSres/(q-2)) por candidato, com
  PENALTY para candidatos que falham (mesma semântica de simulate_model).

O modo de compatibilidade com o notebook NÃO muda: o calibrador só usa o
lote quando recebe ``eval_batch``, e ``optimization.run_calibration``
re-avalia o melhor candidato com ``simulate_model`` (solve_ivp) ao final.

CuPy é opcional: sem ele (ou sem GPU) ``cuda_available()`` retorna False e
a GUI oferece apenas o backend de referência.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

import single_wiebe as sw

_LIMIT = 1.0e12            # mesmo limite de sanidade de single_wiebe._make_rhs
_THREADS = 128
_DTYPES = {"float64": (np.float64, "double"), "float32": (np.float32, "float")}


# =============================================================================
# Detecção
# =============================================================================
def cuda_available() -> bool:
    """True se CuPy está instalado e há ao menos uma GPU CUDA utilizável."""
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def gpu_name() -> Optional[str]:
    """Nome da GPU 0 (ou None se indisponível)."""
    if not cuda_available():
        return None
    import cupy
    nome = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    return nome.decode("utf-8", "replace") if isinstance(nome, bytes) else str(nome)


def _consts(cfg: sw.EngineConfig) -> dict:
    return dict(
        Vd=cfg.Vd, R=cfg.R_rod_ratio, bore=cfg.bore, stroke=cfg.stroke,
        rod=cfg.rod_length, kappa=cfg.kappa, T1=cfg.T1, Tw=cfg.Tw,
        Qtot=cfg.m_comb * cfg.pci, a=cfg.a_wiebe,
        Vp_fac=(cfg.Vp + 1.4) ** 0.8, om2pi=2.0 * math.pi * cfg.omega_rev_s,
        heat=bool(cfg.heat_transfer),
    )


# =============================================================================
# RK4 em lote — NumPy (referência do integrador em lote, CPU)
# =============================================================================
def batch_rk4_numpy(theta_exp: np.ndarray, P1: float, X: np.ndarray,
                    cfg: Optional[sw.EngineConfig] = None,
                    substeps: int = 4) -> np.ndarray:
    """Integra N candidatos (N,4) com RK4 de passo fixo, vetorizado.

    Retorna P_sim (N, n) em kPa; linha com NaN = candidato falho (RHS não
    finita ou > 1e12, P/Tg não positivos) — mesma semântica de
    ``simulate_model``.
    """
    c = _consts(cfg or sw.DEFAULT_CONFIG)
    theta_exp = np.asarray(theta_exp, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    N, n = X.shape[0], theta_exp.size
    Rc, m, th0, dth = (X[:, j] for j in range(4))
    Vd, R, bore, stroke, rod = c["Vd"], c["R"], c["bore"], c["stroke"], c["rod"]
    r_cr = stroke / 2.0
    A_head = 2.0 * math.pi * (bore / 2.0) ** 2
    with np.errstate(all="ignore"):
        Vc = Vd / (Rc - 1.0)
        A_cyl = math.pi * bore * stroke / (Rc - 1.0)
        t0 = float(theta_exp[0])
        V1 = Vc + (Vd / 2.0) * (R + 1.0 - math.cos(t0)
                                - math.sqrt(R * R - math.sin(t0) ** 2))
        dTg0 = c["T1"] / (P1 * V1)
    valido = (Rc > 1.0) & (dth > 0.0)

    def rhs(th, P, Tg):
        s, co = math.sin(th), math.cos(th)
        rootR = math.sqrt(R * R - s * s)
        V = Vc + (Vd / 2.0) * (R + 1.0 - co - rootR)
        dV = (Vd * s / 2.0) * (1.0 + co / rootR)
        y = rod + r_cr - r_cr * co - math.sqrt(rod * rod - r_cr * r_cr * s * s)
        As = A_head + math.pi * bore * y + A_cyl
        ativo = th >= th0
        z = np.where(ativo, np.maximum((th - th0) / dth, 0.0), 0.0)
        dx = c["a"] * (m + 1.0) / dth * z ** m * np.exp(-c["a"] * z ** (m + 1.0))
        dQ = c["Qtot"] * np.where(ativo, dx, 0.0)
        if c["heat"]:
            h = (130.0 * V ** (-0.06) * (P * 1.0e-2) ** 0.8 * Tg ** (-0.4)
                 * c["Vp_fac"])
            dQp = h * As * (Tg - c["Tw"]) / c["om2pi"]
        else:
            dQp = np.zeros(N)
        dP = (1.0 / V) * ((c["kappa"] - 1.0) * (dQ - dQp / 1000.0)
                          - c["kappa"] * P * dV)
        dTg = dTg0 * (V * dP + P * dV)
        return dP, dTg, dQp

    def ok(k):
        return np.all([np.isfinite(v) & (np.abs(v) <= _LIMIT) for v in k],
                      axis=0)

    alive = valido.copy()
    P = np.full(N, float(P1))
    Tg = np.full(N, c["T1"])
    out = np.full((N, n), np.nan)
    out[:, 0] = P1
    with np.errstate(all="ignore"):
        for k in range(n - 1):
            a = float(theta_exp[k])
            h = (float(theta_exp[k + 1]) - a) / substeps
            for _ in range(substeps):
                k1 = rhs(a, P, Tg)
                alive &= ok(k1)
                k2 = rhs(a + h / 2, P + h / 2 * k1[0], Tg + h / 2 * k1[1])
                alive &= ok(k2)
                k3 = rhs(a + h / 2, P + h / 2 * k2[0], Tg + h / 2 * k2[1])
                alive &= ok(k3)
                k4 = rhs(a + h, P + h * k3[0], Tg + h * k3[1])
                alive &= ok(k4)
                P = np.where(alive, P + h / 6 * (k1[0] + 2 * k2[0]
                                                 + 2 * k3[0] + k4[0]), P)
                Tg = np.where(alive, Tg + h / 6 * (k1[1] + 2 * k2[1]
                                                   + 2 * k3[1] + k4[1]), Tg)
                alive &= (P > 0.0) & (Tg > 0.0) & np.isfinite(P) & np.isfinite(Tg)
                a += h
            out[:, k + 1] = P
    out[~alive] = np.nan
    return out


# =============================================================================
# RK4 em lote — CUDA (CuPy)
# =============================================================================
_SOURCE = r"""
#define RL(x) ((real)(x))
#define PI RL(3.141592653589793)

__device__ __forceinline__ bool ok3(real a, real b, real c) {
    const real L = RL(1.0e12);
    return isfinite(a) && isfinite(b) && isfinite(c)
        && fabs(a) <= L && fabs(b) <= L && fabs(c) <= L;
}

// RHS de um candidato — idêntico a single_wiebe._make_rhs
__device__ __forceinline__ void rhs(
        real th, real P, real Tg, real Vc, real A_cyl, real A_head,
        real Vd, real R, real bore, real rod, real r_cr, real kappa,
        real Tw, real Qtot, real a_w, real Vp_fac, real om2pi, int heat,
        real m, real th0, real dth, real dTg0,
        real* dP, real* dTg, real* dQp) {
    const real s = sin(th), c = cos(th);
    const real rootR = sqrt(R * R - s * s);
    const real V = Vc + (Vd / RL(2.0)) * (R + RL(1.0) - c - rootR);
    const real dV = (Vd * s / RL(2.0)) * (RL(1.0) + c / rootR);
    const real y = rod + r_cr - r_cr * c - sqrt(rod * rod - r_cr * r_cr * s * s);
    const real As = A_head + PI * bore * y + A_cyl;
    real dQ = RL(0.0);
    if (th >= th0) {
        real z = (th - th0) / dth;
        if (z < RL(0.0)) z = RL(0.0);
        dQ = Qtot * (a_w * (m + RL(1.0)) / dth * pow(z, m)
                     * exp(-a_w * pow(z, m + RL(1.0))));
    }
    real qp = RL(0.0);
    if (heat) {
        const real h = RL(130.0) * pow(V, RL(-0.06))
            * pow(P * RL(1.0e-2), RL(0.8)) * pow(Tg, RL(-0.4)) * Vp_fac;
        qp = h * As * (Tg - Tw) / om2pi;
    }
    *dP = (RL(1.0) / V) * ((kappa - RL(1.0)) * (dQ - qp / RL(1000.0))
                           - kappa * P * dV);
    *dTg = dTg0 * (V * *dP + P * dV);
    *dQp = qp;
}

extern "C" __global__ void rk4_single_wiebe(
        const double* theta_exp, const real P1, const real* X,
        const int N, const int n,
        const real Vd, const real R, const real bore, const real stroke,
        const real rod, const real kappa, const real T1, const real Tw,
        const real Qtot, const real a_w, const real Vp_fac, const real om2pi,
        const int heat, const int substeps, double* P_out) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    double* row = P_out + (size_t)i * n;
    const real Rc = X[4 * i], m = X[4 * i + 1];
    const real th0 = X[4 * i + 2], dth = X[4 * i + 3];
    if (!(Rc > RL(1.0)) || !(dth > RL(0.0))) return;

    const real r_cr = stroke / RL(2.0);
    const real A_head = RL(2.0) * PI * (bore / RL(2.0)) * (bore / RL(2.0));
    const real Vc = Vd / (Rc - RL(1.0));
    const real A_cyl = PI * bore * stroke / (Rc - RL(1.0));
    const real t0 = (real)theta_exp[0];
    const real V1 = Vc + (Vd / RL(2.0))
        * (R + RL(1.0) - cos(t0) - sqrt(R * R - sin(t0) * sin(t0)));
    const real dTg0 = T1 / (P1 * V1);

    real P = P1, Tg = T1;
    row[0] = (double)P;
    for (int k = 0; k < n - 1; ++k) {
        real a = (real)theta_exp[k];
        const real h = ((real)theta_exp[k + 1] - a) / (real)substeps;
        const real h2 = h / RL(2.0);
        for (int s = 0; s < substeps; ++s) {
            real k1p, k1t, k1w, k2p, k2t, k2w, k3p, k3t, k3w, k4p, k4t, k4w;
            rhs(a, P, Tg, Vc, A_cyl, A_head, Vd, R, bore, rod, r_cr, kappa,
                Tw, Qtot, a_w, Vp_fac, om2pi, heat, m, th0, dth, dTg0,
                &k1p, &k1t, &k1w);
            if (!ok3(k1p, k1t, k1w)) return;
            rhs(a + h2, P + h2 * k1p, Tg + h2 * k1t, Vc, A_cyl, A_head, Vd,
                R, bore, rod, r_cr, kappa, Tw, Qtot, a_w, Vp_fac, om2pi,
                heat, m, th0, dth, dTg0, &k2p, &k2t, &k2w);
            if (!ok3(k2p, k2t, k2w)) return;
            rhs(a + h2, P + h2 * k2p, Tg + h2 * k2t, Vc, A_cyl, A_head, Vd,
                R, bore, rod, r_cr, kappa, Tw, Qtot, a_w, Vp_fac, om2pi,
                heat, m, th0, dth, dTg0, &k3p, &k3t, &k3w);
            if (!ok3(k3p, k3t, k3w)) return;
            rhs(a + h, P + h * k3p, Tg + h * k3t, Vc, A_cyl, A_head, Vd, R,
                bore, rod, r_cr, kappa, Tw, Qtot, a_w, Vp_fac, om2pi, heat,
                m, th0, dth, dTg0, &k4p, &k4t, &k4w);
            if (!ok3(k4p, k4t, k4w)) return;
            P = P + (h / RL(6.0)) * (k1p + RL(2.0) * k2p + RL(2.0) * k3p + k4p);
            Tg = Tg + (h / RL(6.0)) * (k1t + RL(2.0) * k2t + RL(2.0) * k3t + k4t);
            a = a + h;
            if (!(P > RL(0.0)) || !(Tg > RL(0.0))
                    || !isfinite(P) || !isfinite(Tg)) return;
        }
        row[k + 1] = (double)P;
    }
}
"""

_KERNELS: dict = {}


def _kernel(precision: str):
    """Compila (uma vez por processo e precisão) o kernel RK4 em lote."""
    if precision not in _KERNELS:
        import cupy as cp
        src = f"typedef {_DTYPES[precision][1]} real;\n" + _SOURCE
        _KERNELS[precision] = cp.RawKernel(src, "rk4_single_wiebe",
                                           options=("-fmad=false",))
    return _KERNELS[precision]


def _cuda_device(theta_d, P1: float, X: np.ndarray, cfg, substeps: int,
                 precision: str):
    """Roda o kernel; retorna cupy (N, n) float64 (linha com NaN = falho)."""
    import cupy as cp

    if precision not in _DTYPES:
        raise ValueError(f"precision deve ser uma de {sorted(_DTYPES)}.")
    dt = _DTYPES[precision][0]
    c = _consts(cfg or sw.DEFAULT_CONFIG)
    X = np.ascontiguousarray(np.atleast_2d(X), dtype=dt)
    N, n = X.shape[0], int(theta_d.size)
    out = cp.full((N, n), cp.nan, dtype=cp.float64)
    if N == 0:
        return out
    args = (theta_d, dt(P1), cp.asarray(X), np.int32(N), np.int32(n),
            *(dt(c[k]) for k in ("Vd", "R", "bore", "stroke", "rod", "kappa",
                                 "T1", "Tw", "Qtot", "a", "Vp_fac", "om2pi")),
            np.int32(1 if c["heat"] else 0), np.int32(substeps), out)
    _kernel(precision)(((N + _THREADS - 1) // _THREADS,), (_THREADS,), args)
    return out


def batch_rk4_cuda(theta_exp: np.ndarray, P1: float, X: np.ndarray,
                   cfg: Optional[sw.EngineConfig] = None, substeps: int = 4,
                   precision: str = "float64") -> np.ndarray:
    """Mesmo contrato de ``batch_rk4_numpy``, executado na GPU."""
    import cupy as cp
    theta_d = cp.asarray(np.ascontiguousarray(theta_exp, dtype=np.float64))
    return cp.asnumpy(_cuda_device(theta_d, P1, X, cfg, substeps, precision))


# =============================================================================
# Função objetivo em lote
# =============================================================================
BACKENDS = ("cpu", "cuda")


def make_batch_objective(theta_exp: np.ndarray, pressure_exp: np.ndarray,
                         cfg: Optional[sw.EngineConfig] = None,
                         backend: str = "cuda", precision: str = "float64",
                         substeps: int = 4) -> Callable[[np.ndarray], np.ndarray]:
    """Retorna f(X) -> erros (N,), erro = sqrt(SSres/(q-2)) (compute_error),
    PENALTY para candidatos falhos. X: (N, 4) = [Rc, m, theta0, delta_theta].

    backend "cuda": dados experimentais ficam na GPU e o erro é reduzido lá
    (por chamada sobem N x 4 valores e descem N). backend "cpu": RK4 em lote
    NumPy (mesmo integrador, sem GPU).
    """
    theta_exp = np.ascontiguousarray(theta_exp, dtype=np.float64)
    P_exp = np.ascontiguousarray(pressure_exp, dtype=np.float64)
    P1, q = float(P_exp[0]), P_exp.size
    if backend == "cpu":
        def f_cpu(X):
            P_sim = batch_rk4_numpy(theta_exp, P1, X, cfg, substeps)
            ok = np.isfinite(P_sim).all(axis=1)
            err = np.full(P_sim.shape[0], sw.PENALTY)
            d = P_exp[None, :] - P_sim[ok]
            err[ok] = np.sqrt(np.sum(d * d, axis=1) / (q - 2.0))
            return err
        return f_cpu
    if backend != "cuda":
        raise ValueError(f"backend deve ser um de {BACKENDS}.")
    if not cuda_available():
        raise RuntimeError("GPU CUDA indisponível: instale CuPy "
                           "(pip install cupy-cuda12x) e verifique o driver.")
    import cupy as cp
    theta_d, P_exp_d = cp.asarray(theta_exp), cp.asarray(P_exp)

    def f_cuda(X):
        P_sim = _cuda_device(theta_d, P1, X, cfg, substeps, precision)
        ok = cp.isfinite(P_sim).all(axis=1)
        d = P_exp_d[None, :] - P_sim
        err = cp.where(ok, cp.sqrt(cp.sum(d * d, axis=1) / (q - 2.0)),
                       sw.PENALTY)
        return cp.asnumpy(err)
    return f_cuda
