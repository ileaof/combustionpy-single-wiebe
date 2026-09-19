# -*- coding: utf-8 -*-
"""
optimization.py
===============================================================================
Camada de otimização da GUI Single Wiebe.

Orquestra os calibradores de single_wiebe.py (que mantêm os algoritmos)
adicionando o que a interface precisa:
  * seleção de parâmetros livres (os demais ficam fixos);
  * limites por parâmetro (em rad internamente, graus na interface);
  * refinamento local opcional (L-BFGS-B) após a busca global;
  * alertas de ótimo na borda do domínio;
  * tabela comparativa inicial x calibrado.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

import single_wiebe as sw

# Ordem canônica do vetor de parâmetros [internamente em rad para ângulos]
PARAM_NAMES = ["Rc", "m", "theta0", "delta_theta"]
PARAM_UNITS = ["-", "-", "rad", "rad"]
PARAM_LABELS = {
    "Rc": "Razão de compressão (Rc)",
    "m": "Fator de forma (m)",
    "theta0": "Início da combustão (θ₀)",
    "delta_theta": "Duração da combustão (Δθ)",
}

# Limites padrão (idênticos ao notebook)
DEFAULT_BOUNDS = {
    "Rc": (15.0, 17.0),
    "m": (0.1, 1.0),
    "theta0": (np.radians(-1.0), np.radians(2.0)),
    "delta_theta": (np.radians(40.0), np.radians(60.0)),
}


class OptimizationError(Exception):
    """Erro de configuração da otimização com mensagem amigável."""


def build_bounds(selected: List[str],
                 bounds: Dict[str, tuple]) -> (np.ndarray, np.ndarray):
    """Monta vetores lower/upper na ordem PARAM_NAMES para os parâmetros
    selecionados (os não selecionados são removidos do problema)."""
    lower = np.array([bounds[p][0] for p in selected], dtype=float)
    upper = np.array([bounds[p][1] for p in selected], dtype=float)
    if np.any(lower >= upper):
        bad = [PARAM_NAMES[i] for i in range(len(selected)) if lower[i] >= upper[i]]
        raise OptimizationError(
            f"Limite inferior deve ser menor que o superior em: {', '.join(bad)}."
        )
    return lower, upper


def _expand_to_full(x_sel: np.ndarray, selected: List[str],
                    fixed: Dict[str, float]) -> np.ndarray:
    """Reconstrói o vetor completo [Rc, m, theta0, delta_theta] a partir do
    subvetor dos parâmetros livres, inserindo os valores fixos."""
    full = np.array([fixed[p] for p in PARAM_NAMES], dtype=float)
    for i, p in enumerate(selected):
        full[PARAM_NAMES.index(p)] = x_sel[i]
    return full


def run_calibration(
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    selected: List[str],
    bounds: Dict[str, tuple],
    x_init: Dict[str, float],
    cfg: sw.EngineConfig,
    method: str = "PSO",
    n_particulas: int = 20,
    maxiter: int = 300,
    tolerancia: float = 1e-10,
    seed: Optional[int] = None,
    local_polish: bool = False,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict:
    """Executa a calibração com os parâmetros selecionados.

    Retorna dicionário com: params completos, erro, histórico, tabela,
    alertas de borda e flag de cancelamento. Não lança exceção em falha
    de integração (a penalidade do modelo cuida disso).
    """
    if not selected:
        raise OptimizationError("Selecione pelo menos um parâmetro para calibrar.")
    lower, upper = build_bounds(selected, bounds)
    fixed = {p: float(x_init[p]) for p in PARAM_NAMES}

    # Valores iniciais dos livres devem respeitar os limites
    x0_sel = np.array([float(x_init[p]) for p in selected])
    x0_sel = np.clip(x0_sel, lower, upper)

    def f_full(x_sel: np.ndarray) -> float:
        x = _expand_to_full(x_sel, selected, fixed)
        return sw.simulate_model(*x, theta_exp, pressure_exp, cfg=cfg)[0]

    def expand(x_sel: np.ndarray) -> np.ndarray:
        return _expand_to_full(x_sel, selected, fixed)

    if method == "PSO":
        res = sw.calibrate_pso_bounds(
            theta_exp, pressure_exp, lower, upper, seed=seed,
            n_particulas=n_particulas, max_iteracoes=maxiter,
            tolerancia=tolerancia, cfg=cfg,
            progress_callback=progress_callback, cancel_check=cancel_check,
            expand=expand,
        )
    elif method == "DE":
        res = sw.calibrate_de_bounds(
            theta_exp, pressure_exp, lower, upper, seed=seed,
            maxiter=maxiter, popsize=max(5, n_particulas // 2),
            tol=tolerancia, cfg=cfg,
            progress_callback=progress_callback, cancel_check=cancel_check,
            expand=expand,
        )
    else:
        raise OptimizationError(f"Método desconhecido: {method}")

    x_best_sel = np.clip(np.asarray(res["params"], dtype=float), lower, upper)
    erro_best = float(res["erro"])

    # Refinamento local opcional (não faz parte do modo de compatibilidade)
    polish_info = None
    if local_polish and not res.get("cancelado"):
        try:
            loc = minimize(
                f_full, x_best_sel, method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": 200},
            )
            if np.isfinite(loc.fun) and loc.fun < erro_best:
                polish_info = {"erro_antes": erro_best, "erro_depois": float(loc.fun)}
                x_best_sel = np.clip(loc.x, lower, upper)
                erro_best = float(loc.fun)
        except Exception:
            pass  # polish é opcional; mantém o resultado global

    x_full = _expand_to_full(x_best_sel, selected, fixed)

    # Tabela comparativa e alertas de borda
    tabela, alertas = [], []
    for i, p in enumerate(PARAM_NAMES):
        lo, hi = bounds[p]
        val = float(x_full[i])
        linha = {
            "Parâmetro": PARAM_LABELS[p],
            "Valor inicial": float(x_init[p]),
            "Valor calibrado": val,
            "Limite inferior": lo,
            "Limite superior": hi,
            "Unidade": "rad" if p in ("theta0", "delta_theta") else "-",
        }
        tabela.append(linha)
        if p in selected:
            eps = 1e-6 * max(1.0, abs(hi - lo))
            if abs(val - lo) < eps or abs(val - hi) < eps:
                alertas.append(
                    f"⚠ {PARAM_LABELS[p]} convergiu para a borda do domínio "
                    f"({'inferior' if abs(val - lo) < eps else 'superior'}) "
                    f"— o ótimo pode estar fora dos limites."
                )

    return {
        "params": x_full,
        "selected": list(selected),
        "erro": erro_best,
        "iteracoes": int(res.get("iteracoes", 0)),
        "history": res.get("history", []),
        "history_params": res.get("history_params", []),
        "parou_por_repeticao": res.get("parou_por_repeticao", False),
        "cancelado": bool(res.get("cancelado", False)),
        "polish": polish_info,
        "tabela": tabela,
        "alertas": alertas,
    }