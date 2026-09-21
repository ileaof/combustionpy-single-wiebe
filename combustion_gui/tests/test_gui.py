# -*- coding: utf-8 -*-
"""Testes da GUI Single Wiebe (pytest).

Executar de combustion_gui/:
    python -m pytest tests/ -v
"""

import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_processing as dp
import optimization as opt
import plotting as pl
import reporting as rep
import single_wiebe as sw

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "P_exp-Carga-3_45%.txt"

VALID = (17.0, 0.504, math.radians(-6.54), math.radians(62.0))


@pytest.fixture(scope="module")
def data():
    theta, P, resumo = dp.load_file_full(
        SAMPLE, angle_col=0, pressure_col=1, angle_unit="radianos",
        pressure_unit="bar", theta_min=-2.0, theta_max=2.0,
    )
    return theta, P, resumo


@pytest.fixture(scope="module")
def res(data):
    theta, P, _ = data
    return sw.simulate_full(*VALID, theta, P)


# ---------------------------------------------------------------------------
# Leitura e conversão de unidades
# ---------------------------------------------------------------------------
def test_leitura_459_pontos(data):
    theta, P, resumo = data
    assert theta.size == 459 and P.size == 459
    assert resumo["n_obs"] == 459


def test_conversao_unidades():
    df = pd.DataFrame({0: [0.0, 90.0, 180.0], 1: [1.0, 2.0, 3.0]})
    theta, P, _ = dp.prepare_series(
        df, angle_col=0, pressure_col=1, angle_unit="graus",
        pressure_unit="bar", min_points=2)
    assert np.isclose(theta[1], math.pi / 2)
    assert np.allclose(P, [100.0, 200.0, 300.0])  # bar -> kPa


def test_conversao_pa():
    df = pd.DataFrame({0: [0.0], 1: [138200.0]})
    _, P, _ = dp.prepare_series(df, 0, 1, "radianos", "Pa", min_points=1)
    assert abs(P[0] - 138.2) < 1e-9


def test_rejeita_arquivo_de_uma_coluna():
    df = pd.DataFrame({0: [1.0, 2.0]})
    with pytest.raises(Exception):
        dp.read_table(io.StringIO("1.0\n2.0\n"), sep="auto")


def test_angulos_fora_de_ordem_erro():
    df = pd.DataFrame({0: [0.5, 0.1, 0.3], 1: [100.0, 200.0, 300.0]})
    with pytest.raises(dp.DataError):
        dp.prepare_series(df, 0, 1, "radianos", "kPa", sort_by_angle=False,
                          min_points=2)


# ---------------------------------------------------------------------------
# Geometria (EngineConfig)
# ---------------------------------------------------------------------------
def test_geometria_default_igual_constantes():
    cfg = sw.EngineConfig()
    assert abs(cfg.Vd - sw.VD) < 1e-18
    assert abs(cfg.A_cyl - sw.A_CYL) < 1e-18
    assert abs(cfg.Vp - sw.VP) < 1e-15
    assert abs(cfg.omega_rev_s - sw.OMEGA_REV_S) < 1e-15


def test_geometria_custom():
    cfg = sw.EngineConfig(bore=0.1, stroke=0.1, rod_length=0.2, rpm=6000.0)
    assert abs(cfg.A_cyl - math.pi * 0.05**2) < 1e-15
    assert abs(cfg.Vd - cfg.A_cyl * 0.1) < 1e-18
    assert abs(cfg.omega_rev_s - 100.0) < 1e-12


def test_validacao_config_invalida():
    bad = sw.EngineConfig(bore=-0.1, stroke=0.07)
    erros = bad.validate()
    assert any("diâmetro" in e for e in erros)
    bad2 = sw.EngineConfig(rod_length=0.03)  # biela <= s/2 (0.035)
    assert any("Biela" in e for e in bad2.validate())


# ---------------------------------------------------------------------------
# Função de Wiebe
# ---------------------------------------------------------------------------
def test_wiebe_zero_antes_de_theta0():
    x, dx = sw.burned_fraction(np.array([-1.0, -0.1]), 0.0, 1.0, 0.5)
    assert np.all(x == 0) and np.all(dx == 0)


def test_wiebe_com_cfg_custom():
    # a_wiebe diferente altera o resultado
    cfg = sw.EngineConfig(a_wiebe=3.0)
    x1, _ = sw.burned_fraction(0.5, 0.0, 1.0, 0.5)
    x2, _ = sw.burned_fraction(0.5, 0.0, 1.0, 0.5, cfg)
    assert x2 < x1  # a menor -> queima mais lenta


def test_calor_liberado_com_cfg():
    cfg = sw.EngineConfig(m_comb=2e-6, pci=40000.0)
    _, dx = sw.heat_release(0.5, 0.0, 1.0, 0.5, cfg)
    _, dx_def = sw.heat_release(0.5, 0.0, 1.0, 0.5)
    assert dx < dx_def


# ---------------------------------------------------------------------------
# EDOs
# ---------------------------------------------------------------------------
def test_edo_validacao_notebook(data):
    theta, P, _ = data
    erro, Ps, Tg, Qp = sw.simulate_model(*VALID, theta, P)
    assert abs(erro - 409.385) < 0.1
    assert Ps.shape == theta.shape and np.all(np.isfinite(Ps))


def test_edo_sem_transferencia_de_calor(data):
    theta, P, _ = data
    cfg = sw.EngineConfig(heat_transfer=False)
    erro_nt, P_nt, _, Qp_nt = sw.simulate_model(*VALID, theta, P, cfg=cfg)
    erro_h, _, _, Qp_h = sw.simulate_model(*VALID, theta, P)
    assert erro_nt != erro_h
    assert np.all(Qp_nt == 0) and Qp_h[-1] > 0


def test_edo_penalidade(data):
    theta, P, _ = data
    erro, Ps, _, _ = sw.simulate_model(16.0, -2.0, theta[0], 1.0, theta, P)
    assert erro >= sw.PENALTY and Ps is None


# ---------------------------------------------------------------------------
# Métricas de ajuste
# ---------------------------------------------------------------------------
def test_metricas_perfeitas():
    v = np.array([100.0, 200.0, 300.0])
    assert sw.rmse(v, v) == 0.0
    assert abs(sw.r_squared(v, v) - 1.0) < 1e-12


def test_metricas_valores_conhecidos():
    P = np.array([100.0, 200.0, 300.0, 400.0])
    Q = np.array([110.0, 190.0, 310.0, 390.0])
    assert abs(sw.rmse(P, Q) - 10.0) < 1e-12
    assert abs(sw.r_squared(P, Q) - 0.992) < 1e-12  # SSres=400, SStot=50000


# ---------------------------------------------------------------------------
# Otimização (limites, seleção, cancelamento)
# ---------------------------------------------------------------------------
def test_build_bounds_erro():
    with pytest.raises(opt.OptimizationError):
        opt.build_bounds(["Rc"], {"Rc": (17.0, 15.0)})


def test_pso_bounds_respeita_limites(data):
    theta, P, _ = data
    lo = np.array([15.5, 0.3, np.radians(-1.0), np.radians(40.0)])
    hi = np.array([16.5, 0.8, np.radians(2.0), np.radians(60.0)])
    r = sw.calibrate_pso_bounds(theta, P, lo, hi, seed=1, n_particulas=4,
                                max_iteracoes=3)
    assert np.all(r["params"] >= lo - 1e-12)
    assert np.all(r["params"] <= hi + 1e-12)
    assert r["erro"] < sw.PENALTY


def test_pso_bounds_cancelamento(data):
    theta, P, _ = data
    r = sw.calibrate_pso_bounds(
        theta, P, sw.LOWER, sw.UPPER, seed=1, n_particulas=4,
        max_iteracoes=100, cancel_check=lambda: True)
    assert r["cancelado"] and r["iteracoes"] <= 1


def test_de_bounds_roda_e_cancela(data):
    """DE (serial) roda com o scipy instalado e respeita o cancelamento
    (regressão: callback com assinatura que o scipy >= 1.12 não aceitava)."""
    theta, P, _ = data
    r = sw.calibrate_de_bounds(theta, P, sw.LOWER, sw.UPPER, seed=1,
                               maxiter=2, popsize=3)
    assert len(r["history"]) >= 1 and not r["cancelado"]
    assert np.all(r["params"] >= sw.LOWER) and np.all(r["params"] <= sw.UPPER)
    r2 = sw.calibrate_de_bounds(theta, P, sw.LOWER, sw.UPPER, seed=1,
                                maxiter=50, popsize=3,
                                cancel_check=lambda: True)
    assert r2["cancelado"] and len(r2["history"]) == 1


def test_selecao_de_parametros_fixa_os_outros(data):
    theta, P, _ = data
    r = opt.run_calibration(
        theta, P, selected=["m"],
        bounds=opt.DEFAULT_BOUNDS,
        x_init={"Rc": 17.0, "m": 0.5, "theta0": np.radians(-6.54),
                "delta_theta": np.radians(62.0)},
        cfg=sw.EngineConfig(), method="PSO",
        n_particulas=4, maxiter=3, seed=1,
    )
    assert r["params"][0] == 17.0          # Rc fixo
    assert r["params"][2] == np.radians(-6.54)  # theta0 fixo
    assert opt.DEFAULT_BOUNDS["m"][0] <= r["params"][1] <= opt.DEFAULT_BOUNDS["m"][1]


# ---------------------------------------------------------------------------
# Exportações
# ---------------------------------------------------------------------------
def test_csv_export_colunas(res):
    df = pd.read_csv(io.BytesIO(rep.results_csv_bytes(res)), sep=";")
    assert list(df.columns) == rep.GUI_CSV_COLUMNS
    assert len(df) == res.theta.size
    assert np.all(np.isfinite(df["pressure_sim_kPa"]))


def test_json_export(res):
    cfg = sw.EngineConfig()
    b = rep.params_json_bytes(res, cfg, {"rotação [rpm]": cfg.rpm},
                              {"RMSE_kPa": 10.0}, "teste.txt")
    doc = json.loads(b)
    assert doc["analise"]["arquivo_experimental"] == "teste.txt"
    assert doc["combustao"]["Rc"] == 17.0
    assert abs(doc["combustao"]["theta0_deg"] + 6.54) < 1e-9


def test_history_export():
    b = rep.history_csv_bytes([100.0, 80.0], [np.array([16.0, 0.5, 0.0, 1.0]),
                                              np.array([15.9, 0.52, 0.01, 0.98])])
    df = pd.read_csv(io.BytesIO(b), sep=";")
    assert list(df["erro_kPa"]) == [100.0, 80.0]
    assert "Rc_-" in df.columns


def test_relatorio_html(res):
    cfg = sw.EngineConfig()
    figs = pl.export_static_plots(res)
    pngs = {}
    for nome, fig in figs.items():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        pngs[nome] = buf.getvalue()
    html = rep.report_html_bytes(res, cfg, {"x": 1.0}, {"R2": 0.99},
                                 data_name="sample.txt", figures_png=pngs)
    text = html.decode("utf-8")
    assert "<html" in text and "Single Wiebe" in text
    assert "data:image/png;base64," in text


def test_graficos_plotly_nao_vazios(res):
    for fig in (pl.fig_pressure(res), pl.fig_residual(res),
                pl.fig_burned(res), pl.fig_heat_release(res),
                pl.fig_temperature(res), pl.fig_heat_loss(res),
                pl.fig_volume(res), pl.fig_pv(res)):
        assert len(fig.data) >= 1
    assert len(pl.fig_convergence([100.0, 90.0]).data) == 1
    assert len(pl.fig_convergence([]).data) == 0


# ---------------------------------------------------------------------------
# Arquivo de calibração (salvar / abrir)
# ---------------------------------------------------------------------------
def _cal_exemplo():
    return {
        "params": np.array([15.8754, 0.493762, 0.048356, 0.644175]),
        "selected": ["Rc", "m"],
        "erro": 134.6,
        "iteracoes": 42,
        "history": [200.0, 134.6],
        "history_params": [np.array([16.0, 0.5, 0.0, 1.0]),
                           np.array([15.8754, 0.493762, 0.048356, 0.644175])],
        "parou_por_repeticao": False,
        "cancelado": False,
        "polish": {"erro_antes": 135.0, "erro_depois": 134.6},
        "tabela": [{"Parâmetro": "Razão de compressão (Rc)",
                    "Valor inicial": 16.0, "Valor calibrado": 15.8754,
                    "Limite inferior": 15.0, "Limite superior": 17.0,
                    "Unidade": "-"}],
        "alertas": ["⚠ teste"],
    }


def test_calibracao_json_roundtrip():
    cal = _cal_exemplo()
    b = rep.calibration_json_bytes(
        cal, engine_params={"rotação [rpm]": 3396.2}, data_name="dado.txt",
        meta={"metodo": "PSO", "seed": 42},
        theta=np.linspace(-2.0, 2.0, 11), pressure=np.linspace(90.0, 200.0, 11),
    )
    aberto = rep.read_calibration_json(io.BytesIO(b))
    c2 = aberto["calibracao"]
    assert np.allclose(c2["params"], cal["params"])
    assert c2["erro"] == pytest.approx(cal["erro"])
    assert c2["selected"] == cal["selected"]
    assert c2["iteracoes"] == cal["iteracoes"]
    assert c2["polish"] == cal["polish"]
    assert c2["tabela"] == cal["tabela"]
    assert c2["alertas"] == cal["alertas"]
    assert np.allclose(c2["history"], cal["history"])
    assert np.allclose(c2["history_params"][1], cal["history_params"][1])
    assert np.allclose(aberto["theta"], np.linspace(-2.0, 2.0, 11))
    assert np.allclose(aberto["pressao"], np.linspace(90.0, 200.0, 11))
    assert aberto["motor"] == {"rotação [rpm]": 3396.2}
    assert aberto["busca"] == {"metodo": "PSO", "seed": 42}
    assert aberto["arquivo_experimental"] == "dado.txt"
    # sem dados embutidos → par completo ausente não restaura nada
    aberto2 = rep.read_calibration_json(io.BytesIO(
        rep.calibration_json_bytes(cal)))
    assert aberto2["theta"] is None and aberto2["pressao"] is None


def test_calibracao_json_rejeita_arquivo_invalido():
    with pytest.raises(ValueError):
        rep.read_calibration_json(io.BytesIO(b'{"qualquer": 1}'))
    with pytest.raises(ValueError):
        rep.read_calibration_json(io.BytesIO(b"nao e json"))
    with pytest.raises(ValueError):
        rep.read_calibration_json(io.BytesIO(
            b'{"analise": {"formato": "single-wiebe-calibracao"}}'))
    # erro_kPa ausente: ValueError (a GUI só captura ValueError)
    with pytest.raises(ValueError):
        rep.read_calibration_json(io.BytesIO(
            b'{"analise": {"formato": "single-wiebe-calibracao"}, '
            b'"calibracao": {"Rc": 17, "m": 0.5, "theta0_rad": 0, '
            b'"delta_theta_rad": 1}}'))