# -*- coding: utf-8 -*-
"""
plotting.py
===============================================================================
Camada de gráficos da GUI Single Wiebe.

Plotly para os gráficos interativos da interface; Matplotlib apenas para as
exportações estáticas (PNG/PDF). Nenhuma equação aqui — os dados vêm do
ModelResult de single_wiebe.py e a geometria das funções do próprio módulo.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import single_wiebe as sw

# Paleta científica consistente (fundo claro)
C_EXP = "#d62728"      # experimental: vermelho
C_SIM = "#2ca02c"      # simulada: verde
C_RES = "#7f7f7f"      # resíduo: cinza
C_BURN = "#1f77b4"     # fração queimada: azul
C_TEMP = "#9467bd"     # temperatura: roxo
C_LOSS = "#17becf"     # perda: ciano
C_VOL = "#8c564b"      # volume: marrom
C_VLINE = "#ff7f0e"    # theta0: laranja

TEMPLATE = "plotly_white"


def _axis_label(angle_unit: str, label: str) -> str:
    """Rótulo de eixo horizontal conforme a unidade escolhida."""
    unit = "rad" if angle_unit == "rad" else "grau (°)"
    return f"{label} [{unit}]"


def _th(theta: np.ndarray, angle_unit: str) -> np.ndarray:
    """Converte o eixo angular para a unidade de exibição."""
    return theta if angle_unit == "rad" else np.degrees(theta)


def _p(pressure_kPa: np.ndarray, p_unit: str) -> np.ndarray:
    """Converte pressão para a unidade de exibição."""
    return pressure_kPa if p_unit == "kPa" else pressure_kPa / 100.0


def _p_label(p_unit: str) -> str:
    return f"[{p_unit}]"


# ---------------------------------------------------------------------------
# 1. Pressão experimental e simulada (gráfico principal)
# ---------------------------------------------------------------------------
def fig_pressure(res: sw.ModelResult, angle_unit: str = "rad",
                 p_unit: str = "kPa") -> go.Figure:
    """Gráfico principal: experimental ('+' vermelhos) × simulada (verde),
    com linha vertical em theta0, faixa da combustão e ponto de P máxima."""
    th = _th(res.theta, angle_unit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=th, y=_p(res.P_exp, p_unit), mode="markers",
        marker=dict(symbol="cross-thin", size=6, line=dict(width=1.4),
                    color=C_EXP),
        name="Experimental"))
    fig.add_trace(go.Scatter(
        x=th, y=_p(res.P_sim, p_unit), mode="lines",
        line=dict(color=C_SIM, width=2.2), name="Simulada (Single Wiebe)"))

    th0 = _th(np.array([res.theta0]), angle_unit)[0]
    th1 = _th(np.array([res.theta0 + res.delta_theta]), angle_unit)[0]
    fig.add_vline(x=th0, line=dict(color=C_VLINE, width=1.4, dash="dash"),
                  annotation_text="θ₀")
    fig.add_vrect(x0=th0, x1=th1, fillcolor=C_VLINE, opacity=0.12,
                  line_width=0, annotation_text="combustão")

    i_max = int(np.argmax(res.P_exp))
    p_max = _p(np.array([res.P_sim[i_max]]), p_unit)[0]
    p_exp_max = _p(np.array([res.P_exp[i_max]]), p_unit)[0]
    y_top = max(p_max, p_exp_max)
    fig.add_trace(go.Scatter(
        x=[th[i_max]], y=[y_top], mode="markers+text",
        marker=dict(symbol="star", size=12, color=C_VLINE),
        text=[f"P máx = {y_top:,.0f}"], textposition="top center",
        textfont=dict(size=11), name="P máxima",
        showlegend=False))

    fig.update_layout(
        template=TEMPLATE,
        title=f"Pressão simulada × experimental — erro = {res.erro:.2f} kPa",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title=f"Cylinder pressure {_p_label(p_unit)}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=70, b=45),
        height=430,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Resíduo da pressão
# ---------------------------------------------------------------------------
def fig_residual(res: sw.ModelResult, angle_unit: str = "rad",
                 p_unit: str = "kPa") -> go.Figure:
    resid = res.P_exp - res.P_sim
    fig = go.Figure(go.Bar(
        x=_th(res.theta, angle_unit), y=_p(resid, p_unit),
        marker_color=C_RES, name="Resíduo"))
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(
        template=TEMPLATE,
        title="Resíduo da pressão (P_exp − P_sim)",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title=f"Resíduo {_p_label(p_unit)}",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Fração de massa queimada
# ---------------------------------------------------------------------------
def fig_burned(res: sw.ModelResult, angle_unit: str = "rad") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=_th(res.theta, angle_unit), y=res.x_b, mode="lines",
        line=dict(color=C_BURN, width=2.2), name="x_b"))
    fig.update_layout(
        template=TEMPLATE,
        title="Fração de massa queimada (Single Wiebe)",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title="Fração queimada x_b [-]",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Taxa de liberação de calor
# ---------------------------------------------------------------------------
def fig_heat_release(res: sw.ModelResult, angle_unit: str = "rad") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=_th(res.theta, angle_unit), y=res.dQ_dtheta, mode="lines",
        line=dict(color=C_SIM, width=2.2), name="dQ/dθ"))
    fig.update_layout(
        template=TEMPLATE,
        title="Taxa de liberação de calor",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title="dQ/dθ [kJ/rad]",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Temperatura do gás
# ---------------------------------------------------------------------------
def fig_temperature(res: sw.ModelResult, angle_unit: str = "rad") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=_th(res.theta, angle_unit), y=res.Tg_sim, mode="lines",
        line=dict(color=C_TEMP, width=2.2), name="Tg"))
    fig.update_layout(
        template=TEMPLATE,
        title="Temperatura do gás",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title="Temperatura [K]",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Calor perdido acumulado
# ---------------------------------------------------------------------------
def fig_heat_loss(res: sw.ModelResult, angle_unit: str = "rad") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=_th(res.theta, angle_unit), y=res.Qp_sim, mode="lines",
        line=dict(color=C_LOSS, width=2.2), name="Qp"))
    fig.update_layout(
        template=TEMPLATE,
        title="Calor perdido acumulado (Hohenberg)",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title="Calor perdido [J]",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Volume do cilindro
# ---------------------------------------------------------------------------
def fig_volume(res: sw.ModelResult, angle_unit: str = "rad",
               cfg: Optional[sw.EngineConfig] = None) -> go.Figure:
    V = sw.cylinder_volume(res.theta, res.Rc, cfg)
    fig = go.Figure(go.Scatter(
        x=_th(res.theta, angle_unit), y=V * 1e6, mode="lines",
        line=dict(color=C_VOL, width=2.2), name="V(θ)"))
    fig.update_layout(
        template=TEMPLATE,
        title="Volume do cilindro",
        xaxis_title=_axis_label(angle_unit, "Crank angle"),
        yaxis_title="Volume [cm³]",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Diagrama pressão-volume
# ---------------------------------------------------------------------------
def fig_pv(res: sw.ModelResult) -> go.Figure:
    V = sw.cylinder_volume(res.theta, res.Rc) * 1e6  # cm³
    fig = go.Figure(go.Scattergl(
        x=V, y=res.P_sim, mode="lines", line=dict(color=C_SIM, width=2),
        name="Simulada"))
    fig.add_trace(go.Scattergl(
        x=V, y=res.P_exp, mode="markers",
        marker=dict(symbol="cross-thin", size=5, line=dict(width=1),
                    color=C_EXP), name="Experimental"))
    fig.update_layout(
        template=TEMPLATE,
        title="Diagrama pressão-volume",
        xaxis_title="Volume [cm³]",
        yaxis_title="Pressão [kPa]",
        margin=dict(l=50, r=20, t=60, b=45), height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Convergência da calibração
# ---------------------------------------------------------------------------
def fig_convergence(history: List[float], method_name: str = "Calibração"
                    ) -> go.Figure:
    fig = go.Figure()
    if history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)), y=history, mode="lines+markers",
            marker=dict(size=4), line=dict(color=C_BURN, width=1.8),
            name="melhor erro"))
    fig.update_layout(
        template=TEMPLATE,
        title=f"Convergência — {method_name}",
        xaxis_title="Iteração",
        yaxis_title="erro = sqrt(SSres/(q−2)) [kPa]",
        yaxis_type="log",
        margin=dict(l=50, r=20, t=60, b=45), height=330,
    )
    return fig


# ---------------------------------------------------------------------------
# Exportações estáticas (Matplotlib — somente PNG/PDF)
# ---------------------------------------------------------------------------
def export_static_plots(res: sw.ModelResult) -> Dict[str, "matplotlib.figure.Figure"]:
    """Gera as figuras estáticas (mesmo estilo do notebook) para exportação.

    Retorna dicionário {nome: figura_matplotlib}; o chamador salva em
    PNG/PDF. Usa apenas dados de ModelResult + funções do módulo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = res.theta
    figs: Dict = {}

    # Pressão exp × sim (estilo notebook)
    f1, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(th, res.P_exp, "r+", ms=4, label="Experimental")
    ax.plot(th, res.P_sim, "g-", lw=1.4, label="Simulada (Single Wiebe)")
    ax.set_xlabel("Ângulo do Eixo de Manivelas [rad]")
    ax.set_ylabel("Pressão [kPa]")
    ax.set_title(f"Pressão simulada x experimental — erro = {res.erro:.2f} kPa")
    ax.legend(loc="lower right")
    f1.tight_layout()
    figs["pressao_exp_sim"] = f1

    # Fração queimada
    f2, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(th, res.x_b, "b-", lw=1.4)
    ax.set_xlabel("Ângulo do Eixo de Manivelas [rad]")
    ax.set_ylabel("Fração queimada x_b [-]")
    ax.set_title("Fração de massa queimada")
    f2.tight_layout()
    figs["fracao_queimada"] = f2

    # Taxa de liberação
    f3, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(th, res.dQ_dtheta, "g-", lw=1.4)
    ax.set_xlabel("Ângulo do Eixo de Manivelas [rad]")
    ax.set_ylabel("dQ/dθ [kJ/rad]")
    ax.set_title("Taxa de liberação de calor")
    f3.tight_layout()
    figs["dQ_dtheta"] = f3

    # Temperatura
    f4, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(th, res.Tg_sim, "m-", lw=1.4)
    ax.set_xlabel("Ângulo do Eixo de Manivelas [rad]")
    ax.set_ylabel("Temperatura [K]")
    ax.set_title("Temperatura do gás")
    f4.tight_layout()
    figs["temperatura"] = f4

    # Perda de calor
    f5, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(th, res.Qp_sim, "c-", lw=1.4)
    ax.set_xlabel("Ângulo do Eixo de Manivelas [rad]")
    ax.set_ylabel("Calor perdido [J]")
    ax.set_title("Calor perdido acumulado (Hohenberg)")
    f5.tight_layout()
    figs["calor_perdido"] = f5

    # P-V
    f6, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sw.cylinder_volume(th, res.Rc) * 1e6, res.P_sim, "g-", lw=1.4,
            label="Simulada")
    ax.plot(sw.cylinder_volume(th, res.Rc) * 1e6, res.P_exp, "r+", ms=3,
            label="Experimental")
    ax.set_xlabel("Volume [cm³]")
    ax.set_ylabel("Pressão [kPa]")
    ax.set_title("Diagrama pressão-volume")
    ax.legend(loc="lower right")
    f6.tight_layout()
    figs["pv"] = f6

    return figs