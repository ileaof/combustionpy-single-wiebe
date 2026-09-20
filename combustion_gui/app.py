# -*- coding: utf-8 -*-
"""
app.py — Single Wiebe Combustion Analysis (Streamlit)
===============================================================================
Interface gráfica do modelo Single Wiebe (single_wiebe.py). A interface NÃO
contém equações: chamadas a single_wiebe (física), data_processing (dados),
optimization (calibração), plotting (gráficos) e reporting (exportação).

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

import data_processing as dp
import optimization as opt
import plotting as pl
import reporting as rep
import single_wiebe as sw

st.set_page_config(
    page_title="Single Wiebe Combustion Analysis",
    page_icon="🔥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# Single Wiebe Combustion Analysis")
st.markdown(
    "This combustion simulation employs a single Wiebe function and was "
    "developed as part of L. Queiroz’s M.Sc. thesis under the supervision "
    "of Prof. I. L. Ferreira."
)
st.divider()

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------
def _init_state():
    defaults = dict(
        data_theta=None,          # np.ndarray rad
        data_press=None,          # np.ndarray kPa
        data_name="",
        data_summary=None,        # dict
        engine_cfg=sw.EngineConfig(),
        sim_result=None,          # sw.ModelResult
        sim_params=None,          # dict Rc/m/theta0_deg/delta_theta_deg
        calib_result=None,        # dict de optimization.run_calibration
        calib_running=False,
        calib_cancel=False,
        calib_progress=None,      # dict compartilhado com a thread
        tabela_unit="rad",        # unidade de exibição da tabela calibrada
        calib_meta=None,          # dict como a busca rodou (método/seed...)
        calib_open_name=None,     # (nome, tamanho) do último arquivo aberto
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

tabs = st.tabs([
    "Experimental Data", "Engine Setup", "Simulation",
    "Calibration", "Results", "Export",
])

HELP_SEPARADOR = "Use 'auto' para detecção automática (vírgula, ponto-e-vírgula, tabulação ou espaços)."
HELP_THETA0 = "Ângulo do virabrequim em que a combustão começa (graus). Valores negativos = antes do PMS."


def _engine_params_dict(cfg: sw.EngineConfig) -> Dict:
    """Snapshot dos parâmetros do motor p/ exports e arquivo de calibração."""
    return {
        "diâmetro [mm]": cfg.bore * 1000.0,
        "curso [mm]": cfg.stroke * 1000.0,
        "biela [mm]": cfg.rod_length * 1000.0,
        "rotação [rpm]": cfg.rpm,
        "kappa [-]": cfg.kappa,
        "m_comb [kg/ciclo]": cfg.m_comb,
        "PCI [kJ/kg]": cfg.pci,
        "T1 [K]": cfg.T1,
        "Tw [K]": cfg.Tw,
    }


# =============================================================================
# Aba 1 — Experimental Data
# =============================================================================
with tabs[0]:
    st.subheader("Experimental Data")
    up = st.file_uploader("Arquivo de pressão experimental (.txt, .csv, .tsv)",
                          type=["txt", "csv", "tsv"])
    if up is not None:
        st.session_state.data_name = up.name

    c1, c2, c3 = st.columns(3)
    with c1:
        sep = st.text_input("Separador", value="auto", help=HELP_SEPARADOR)
        angle_col = st.number_input("Coluna do ângulo", 0, 9, 0)
    with c2:
        angle_unit = st.selectbox("Unidade angular", ["radianos", "graus"], 0)
        pressure_unit = st.selectbox("Unidade da pressão", ["bar", "kPa", "Pa"], 0)
    with c3:
        pressure_col = st.number_input("Coluna da pressão", 0, 9, 1)
        remove_invalid = st.checkbox("Remover linhas inválidas", value=True)
        sort_by_angle = st.checkbox("Ordenar por ângulo", value=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        theta_min = st.number_input("θ mínimo [rad]", -10.0, 10.0, -2.0,
                                    help="Filtro de intervalo angular (em rad).")
    with c5:
        theta_max = st.number_input("θ máximo [rad]", -10.0, 10.0, 2.0)
    with c6:
        smooth = st.checkbox("Aplicar suavização (Savitzky-Golay)", value=False)
        if smooth:
            smooth_window = st.slider("Janela da suavização", 5, 51, 11, step=2)

    if up is not None and st.button("Carregar dados", type="primary"):
        try:
            kwargs = dict(
                angle_col=int(angle_col), pressure_col=int(pressure_col),
                angle_unit=angle_unit, pressure_unit=pressure_unit,
                theta_min=float(theta_min), theta_max=float(theta_max),
                remove_invalid=remove_invalid, sort_by_angle=sort_by_angle,
            )
            if smooth:
                kwargs.update(smooth=True, smooth_window=smooth_window)
            theta, P, resumo = dp.load_file_full(up, sep=sep, **kwargs)
            st.session_state.data_theta = theta
            st.session_state.data_press = P
            st.session_state.data_summary = resumo
        except dp.DataError as e:
            st.error(f"Erro nos dados: {e}")
        except Exception as e:  # noqa: BLE001 — protege a UI de falhas inesperadas
            st.error(f"Falha ao ler o arquivo: {e}")

    if st.session_state.data_theta is not None:
        theta = st.session_state.data_theta
        P = st.session_state.data_press
        resumo = st.session_state.data_summary

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Observações", f"{resumo['n_obs']}")
        k2.metric("θ mín [rad]", f"{resumo['theta_min']:.4f}")
        k3.metric("θ máx [rad]", f"{resumo['theta_max']:.4f}")
        k4.metric("P mín [kPa]", f"{resumo['P_min']:.1f}")
        k5.metric("P máx [kPa]", f"{resumo['P_max']:.1f}")
        st.caption(
            f"Passo angular médio: {resumo['passo_medio']:.6f} rad "
            f"({np.degrees(resumo['passo_medio']):.3f}°) — "
            f"linhas descartadas: {resumo['n_descartadas']}"
        )

        st.dataframe(
            pd.DataFrame({"theta [rad]": theta[:10], "P [kPa]": P[:10]}),
            use_container_width=True, hide_index=True,
        )
        fig = pl.go.Figure()
        fig.add_trace(pl.go.Scatter(
            x=theta, y=P, mode="markers",
            marker=dict(symbol="cross-thin", size=5,
                        line=dict(width=1.2), color=pl.C_EXP),
            name="Experimental"))
        fig.update_layout(
            template=pl.TEMPLATE,
            title="Pressão experimental",
            xaxis_title="Crank angle [rad]",
            yaxis_title="Cylinder pressure [kPa]",
            height=380, margin=dict(l=50, r=20, t=50, b=45),
        )
        st.plotly_chart(fig, use_container_width=True)

        if abs(resumo["n_obs"] - 459) < 2 and resumo["theta_min"] < -1.9:
            st.info("✅ Aproximadamente 459 observações após o filtro "
                    "−2 ≤ θ ≤ 2 rad — compatível com o arquivo original.")
    else:
        st.info("Envie um arquivo e clique em **Carregar dados**. "
                "O arquivo típico tem 2 colunas sem cabeçalho: "
                "ângulo do virabrequim [rad] e pressão [bar].")


# =============================================================================
# Aba 2 — Engine Setup
# =============================================================================
with tabs[1]:
    st.subheader("Engine Setup")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("**Geometria**")
        bore_mm = st.number_input("Diâmetro do cilindro [mm]", 10.0, 500.0, 86.0)
        stroke_mm = st.number_input("Curso do pistão [mm]", 10.0, 500.0, 70.0)
        rod_mm = st.number_input("Comprimento da biela [mm]", 30.0, 1000.0, 117.5)
        n_cyl = st.number_input("Número de cilindros", 1, 24, 1)
    with g2:
        st.markdown("**Operação**")
        rpm = st.number_input("Rotação [rpm]", 100.0, 20000.0, 3396.20)
        Rc_setup = st.number_input("Razão de compressão [-]", 2.0, 30.0, 17.0,
                                   help="Deve ser > 1. Usada na simulação manual.")
    with g3:
        st.markdown("**Combustível**")
        m_comb = st.number_input("Massa de combustível [kg/ciclo]",
                                 1e-9, 1e-3, 9.42754647351e-6, format="%.6e")
        pci = st.number_input("PCI [kJ/kg]", 1000.0, 120000.0, 39191.3)
    with g4:
        st.markdown("**Termodinâmica e troca térmica**")
        kappa = st.number_input("Razão de calores específicos [-]", 1.1, 1.9, 1.37)
        T1 = st.number_input("Temperatura inicial [K]", 200.0, 500.0, 308.15)
        Tw = st.number_input("Temperatura da parede [K]", 300.0, 600.0, 440.0)
        a_wiebe = st.number_input("Constante de Wiebe (a) [-]", 0.5, 20.0, 6.9078)

    cfg_tmp = sw.EngineConfig(
        bore=bore_mm / 1000.0, stroke=stroke_mm / 1000.0,
        rod_length=rod_mm / 1000.0, rpm=rpm, kappa=kappa,
        m_comb=m_comb, pci=pci, a_wiebe=a_wiebe, T1=T1, Tw=Tw,
    )
    erros_cfg = cfg_tmp.validate()
    if erros_cfg:
        for e in erros_cfg:
            st.error(e)
    else:
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Área do pistão", f"{cfg_tmp.A_cyl * 1e4:.2f} cm²")
        d2.metric("Volume deslocado", f"{cfg_tmp.Vd * 1e6:.2f} cm³")
        d3.metric("Volume de folga (Rc=17)",
                  f"{cfg_tmp.Vd / (Rc_setup - 1.0) * 1e6:.2f} cm³")
        d4.metric("Razão biela/manivela", f"{cfg_tmp.R_rod_ratio:.3f}")
        d5.metric("Vel. média pistão", f"{cfg_tmp.Vp:.2f} m/s")
        d6.metric("Rotação", f"{cfg_tmp.omega_rev_s:.2f} rev/s")
        if st.button("Aplicar configuração", type="primary"):
            st.session_state.engine_cfg = cfg_tmp
            st.success("Configuração do motor aplicada.")


# =============================================================================
# Aba 3 — Simulation
# =============================================================================
with tabs[2]:
    st.subheader("Simulation")
    if st.session_state.data_theta is None:
        st.warning("Carregue os dados experimentais na aba **Experimental Data**.")
    s1, s2, s3 = st.columns(3)
    with s1:
        m_sim = st.number_input("Fator de forma, m [-]", 0.05, 3.0, 0.504)
    with s2:
        theta0_sim = st.number_input("Início da combustão, θ₀ [°]", -60.0, 60.0,
                                     -6.54, help=HELP_THETA0)
    with s3:
        dtheta_sim = st.number_input("Duração da combustão, Δθ [°]", 5.0, 180.0, 62.0)

    with st.expander("Opções avançadas"):
        a1, a2, a3 = st.columns(3)
        with a1:
            method_sim = st.selectbox("Método do integrador",
                                      ["DOP853", "LSODA", "RK45"], 0)
            rtol_sim = st.number_input("rtol", 1e-12, 1e-4, 1e-9, format="%.0e")
        with a2:
            atol_sim = st.number_input("atol", 1e-12, 1e-4, 1e-9, format="%.e")
            Rc_sim = st.number_input("Razão de compressão (Rc) [-]", 2.0, 30.0,
                                     17.0)
        with a3:
            heat_transfer_sim = st.checkbox("Transferência de calor (Hohenberg)",
                                            value=True)
            st.caption("Intervalo da simulação = intervalo dos dados "
                       "carregados (θ mínimo a θ máximo).")

    if st.button("Run simulation", type="primary"):
        try:
            theta = st.session_state.data_theta
            P = st.session_state.data_press
            if theta is None:
                st.error("Carregue os dados experimentais antes de simular.")
            else:
                cfg_sim = st.session_state.engine_cfg
                cfg_sim.heat_transfer = heat_transfer_sim
                with st.spinner("Integrando o sistema de EDOs..."):
                    res = sw.simulate_full(
                        float(Rc_sim), float(m_sim),
                        np.radians(theta0_sim), np.radians(dtheta_sim),
                        theta, P, rtol=rtol_sim, atol=atol_sim,
                        method=method_sim, cfg=cfg_sim,
                    )
                if res.P_sim is None:
                    st.error("A integração falhou para esses parâmetros "
                             "(penalidade aplicada). Verifique os valores.")
                else:
                    st.session_state.sim_result = res
                    st.session_state.sim_params = {
                        "Rc": float(Rc_sim), "m": float(m_sim),
                        "theta0_deg": float(theta0_sim),
                        "delta_theta_deg": float(dtheta_sim),
                        "method": method_sim, "rtol": rtol_sim,
                        "atol": atol_sim,
                    }
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro na simulação: {e}")

    res = st.session_state.sim_result
    if res is not None and res.P_sim is not None:
        theta, P_exp = res.theta, res.P_exp
        i_max = int(np.argmax(res.P_sim))
        Q_total = (st.session_state.engine_cfg.m_comb
                   * st.session_state.engine_cfg.pci)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pressão máxima", f"{res.P_sim.max():,.0f} kPa",
                  f"em θ = {np.degrees(theta[i_max]):.2f}°")
        m2.metric("Temperatura máxima", f"{res.Tg_sim.max():,.0f} K")
        m3.metric("x_b final", f"{res.x_b[-1]:.4f}")
        m4.metric("Calor total liberado", f"{Q_total:.3f} kJ")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Calor perdido", f"{res.Qp_sim[-1]:.2f} J")
        m6.metric("RMSE", f"{sw.rmse(P_exp, res.P_sim):.3f} kPa")
        m7.metric("R²", f"{sw.r_squared(P_exp, res.P_sim):.5f}")
        m8.metric("erro (notebook)", f"{res.erro:.3f} kPa")
    elif res is not None:
        st.error("Última simulação falhou — os resultados anteriores válidos "
                 "foram preservados.")


# =============================================================================
# Aba 4 — Calibration
# =============================================================================
def _calib_worker(run_kwargs: Dict, progress: Dict, state_holder: Dict):
    """Thread de calibração: escreve progresso e resultado no holder."""
    def cb(iteracao, fbest, params):
        progress["iteracao"] = iteracao
        progress["fbest"] = fbest
        progress["params"] = params.tolist()
        progress["t"] = time.time()

    def cancel():
        return progress["cancel"]

    try:
        resultado = opt.run_calibration(
            progress_callback=cb, cancel_check=cancel, **run_kwargs
        )
        state_holder["resultado"] = resultado
    except (opt.OptimizationError, Exception) as e:  # noqa: BLE001
        state_holder["erro"] = str(e)


with tabs[3]:
    st.subheader("Calibration")
    if st.session_state.data_theta is None:
        st.warning("Carregue os dados experimentais na aba **Experimental Data**.")

    st.markdown("**Parâmetros a calibrar** (os não marcados ficam fixos nos "
                "valores iniciais):")
    sel1, sel2, sel3, sel4 = st.columns(4)
    sel_Rc = sel1.checkbox("Rc", value=True)
    sel_m = sel2.checkbox("m", value=True)
    sel_th0 = sel3.checkbox("θ₀", value=True)
    sel_dth = sel4.checkbox("Δθ", value=True)
    selected = [p for p, s in zip(opt.PARAM_NAMES,
                                  [sel_Rc, sel_m, sel_th0, sel_dth]) if s]

    st.markdown("**Valores iniciais e limites** (ângulos em graus):")
    b1, b2, b3, b4 = st.columns(4)
    init_Rc = b1.number_input("Rc inicial", 5.0, 30.0, 16.0)
    lo_Rc, hi_Rc = b1.slider("Limites Rc", 5.0, 30.0, (15.0, 17.0))
    init_m = b2.number_input("m inicial", 0.05, 3.0, 0.5)
    lo_m, hi_m = b2.slider("Limites m", 0.05, 3.0, (0.1, 1.0))
    init_th0 = b3.number_input("θ₀ inicial [°]", -60.0, 60.0, 0.0)
    lo_th0, hi_th0 = b3.slider("Limites θ₀ [°]", -60.0, 60.0, (-1.0, 2.0))
    init_dth = b4.number_input("Δθ inicial [°]", 5.0, 180.0, 50.0)
    lo_dth, hi_dth = b4.slider("Limites Δθ [°]", 5.0, 180.0, (40.0, 60.0))

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        method_cal = st.selectbox("Método", ["PSO", "DE"], 0,
                                  help="PSO: compatível com o notebook. "
                                       "DE: differential_evolution (SciPy).")
    with o2:
        pop_cal = st.number_input("População (PSO) / popsize×2 (DE)", 4, 200, 20)
    with o3:
        iters_cal = st.number_input("Máximo de iterações", 10, 5000, 300)
    with o4:
        seed_cal = st.number_input("Semente aleatória", 0, 999999, 42)
    tol_cal = st.number_input("Tolerância de estagnação", 1e-14, 1e-6, 1e-10,
                              format="%.0e")
    polish_cal = st.checkbox("Refinamento local (L-BFGS-B) após a busca global",
                             value=False)

    col_run, col_cancel, col_load = st.columns([1, 1, 2])
    run_btn = col_run.button("Run calibration", type="primary",
                             disabled=st.session_state.calib_running)
    cancel_btn = col_cancel.button("Cancelar calibração",
                                   disabled=not st.session_state.calib_running)

    # Carregar calibração salva — não exige rodar uma calibração antes
    up_cal = col_load.file_uploader(
        "📂 Carregar calibração (.json)", type=["json"], key="up_calib",
        help="Abre um JSON salvo por “⬇ Salvar calibração” e restaura os "
             "parâmetros calibrados sem recalibrar; os dados experimentais "
             "só são restaurados se a sessão estiver vazia.")
    if up_cal is not None:
        try:
            aberto = rep.read_calibration_json(up_cal)
        except ValueError as e:
            st.error(f"Arquivo de calibração inválido: {e}")
        else:
            marca = (up_cal.name, up_cal.size)
            if st.session_state.calib_open_name != marca:
                st.session_state.calib_result = aberto["calibracao"]
                st.session_state.calib_open_name = marca
                if (aberto["theta"] is not None
                        and st.session_state.data_theta is None):
                    th_ab, P_ab = aberto["theta"], aberto["pressao"]
                    st.session_state.data_theta = th_ab
                    st.session_state.data_press = P_ab
                    st.session_state.data_name = (
                        aberto["arquivo_experimental"] or up_cal.name)
                    st.session_state.data_summary = {
                        "n_obs": int(th_ab.size),
                        "theta_min": float(th_ab.min()),
                        "theta_max": float(th_ab.max()),
                        "P_min": float(P_ab.min()),
                        "P_max": float(P_ab.max()),
                        "passo_medio": float(np.mean(np.diff(th_ab))),
                        "n_descartadas": 0,
                    }
                st.rerun()
            st.success(f"Calibração aberta: erro = "
                       f"{aberto['calibracao']['erro']:.6g} kPa.")
            if aberto.get("busca"):
                st.caption("Busca salva no arquivo: "
                           + "; ".join(f"{k}={v}" for k, v in
                                       aberto["busca"].items()))

    if run_btn:
        if st.session_state.data_theta is None:
            st.error("Carregue os dados experimentais antes de calibrar.")
        elif not selected:
            st.error("Selecione pelo menos um parâmetro para calibrar.")
        else:
            progress = {"iteracao": 0, "fbest": None, "params": None,
                        "t_start": time.time(), "cancel": False}
            st.session_state.calib_progress = progress
            st.session_state.calib_cancel = False
            st.session_state.calib_meta = {
                "metodo": method_cal, "selecao": selected,
                "populacao": int(pop_cal), "maxiter": int(iters_cal),
                "tolerancia": float(tol_cal), "seed": int(seed_cal),
                "polish": bool(polish_cal),
            }
            run_kwargs = dict(
                theta_exp=st.session_state.data_theta,
                pressure_exp=st.session_state.data_press,
                selected=selected,
                bounds={
                    "Rc": (lo_Rc, hi_Rc), "m": (lo_m, hi_m),
                    "theta0": (np.radians(lo_th0), np.radians(hi_th0)),
                    "delta_theta": (np.radians(lo_dth), np.radians(hi_dth)),
                },
                x_init={"Rc": init_Rc, "m": init_m,
                        "theta0": np.radians(init_th0),
                        "delta_theta": np.radians(init_dth)},
                cfg=st.session_state.engine_cfg,
                method=method_cal,
                n_particulas=int(pop_cal),
                maxiter=int(iters_cal),
                tolerancia=tol_cal,
                seed=int(seed_cal),
                local_polish=polish_cal,
            )
            holder: Dict = {}
            th_worker = threading.Thread(
                target=_calib_worker, args=(run_kwargs, progress, holder),
                daemon=True)
            th_worker.start()
            st.session_state.calib_running = True
            status = st.empty()
            bar = st.progress(0.0)
            meta = st.empty()
            while th_worker.is_alive():
                status.metric("Iteração", progress["iteracao"],
                              delta=(f"melhor erro = {progress['fbest']:.3f} kPa"
                                     if progress["fbest"] else None))
                if progress["fbest"] and progress["params"] is not None:
                    p = progress["params"]
                    meta.caption(
                        f"melhores parâmetros: Rc={p[0]:.4f}, m={p[1]:.4f}, "
                        f"θ₀={np.degrees(p[2]):.2f}°, Δθ={np.degrees(p[3]):.2f}° "
                        f"| decorrido: {time.time() - progress['t_start']:.0f} s"
                    )
                bar.progress(min(1.0, progress["iteracao"] / max(1.0, float(iters_cal))))
                time.sleep(0.5)
            th_worker.join()
            st.session_state.calib_running = False
            bar.progress(1.0)
            if "erro" in holder:
                st.error(f"Erro na calibração: {holder['erro']}")
            elif "resultado" in holder:
                st.session_state.calib_result = holder["resultado"]
                r = holder["resultado"]
                if r["cancelado"]:
                    st.warning("Calibração cancelada — resultados parciais "
                               "disponíveis abaixo; execuções anteriores "
                               "preservadas.")
                else:
                    st.success(
                        f"Calibração concluída: erro = {r['erro']:.6f} kPa em "
                        f"{r['iteracoes']} iterações."
                    )

    if cancel_btn and st.session_state.calib_running:
        st.session_state.calib_progress["cancel"] = True
        st.info("Cancelamento solicitado — aguardando a iteração atual...")

    cal = st.session_state.calib_result
    if cal is not None:
        st.markdown("#### Parâmetros calibrados")
        # Botão de conversão rad <-> grau: afeta SÓ a exibição da tabela;
        # os valores internos (cal["params"]) permanecem em radianos.
        unidade = st.session_state.tabela_unit
        rotulo = ("Converter tabela para graus (°)" if unidade == "rad"
                  else "Converter tabela para radianos (rad)")
        if st.button(rotulo, key="converter_tabela"):
            st.session_state.tabela_unit = "grau" if unidade == "rad" else "rad"
        unidade = st.session_state.tabela_unit

        # θ₀ e Δθ são guardados em rad; Rc e m são adimensionais
        angulos = {opt.PARAM_LABELS["theta0"], opt.PARAM_LABELS["delta_theta"]}
        tabela_vis = []
        for linha in cal["tabela"]:
            l2 = dict(linha)
            if l2["Parâmetro"] in angulos:
                if unidade == "grau":
                    for k in ("Valor inicial", "Valor calibrado",
                              "Limite inferior", "Limite superior"):
                        l2[k] = np.degrees(l2[k])
                l2["Unidade"] = "grau (°)" if unidade == "grau" else "rad"
            tabela_vis.append(l2)
        st.dataframe(pd.DataFrame(tabela_vis), use_container_width=True,
                     hide_index=True)
        for a in cal["alertas"]:
            st.warning(a)
        if cal.get("polish"):
            st.info(f"Refinamento local: erro {cal['polish']['erro_antes']:.6f} "
                    f"→ {cal['polish']['erro_depois']:.6f} kPa.")

        # Mesmo comportamento do Double Wiebe: aplica os parâmetros calibrados
        # e roda a simulação (os ângulos já chegam em rad, como o modelo espera)
        if st.button("Aplicar parâmetros calibrados e rodar a simulação",
                     key="aplicar_calib"):
            theta = st.session_state.data_theta
            P = st.session_state.data_press
            if theta is None:
                st.error("Carregue os dados experimentais antes de simular.")
            else:
                p = cal["params"]
                opts = st.session_state.sim_params or {}
                cfg_ap = st.session_state.engine_cfg
                try:
                    with st.spinner("Integrando o sistema de EDOs..."):
                        res = sw.simulate_full(
                            float(p[0]), float(p[1]), float(p[2]), float(p[3]),
                            theta, P,
                            rtol=opts.get("rtol", 1e-9),
                            atol=opts.get("atol", 1e-9),
                            method=opts.get("method", "DOP853"),
                            cfg=cfg_ap,
                        )
                    if res.P_sim is None:
                        st.error("A integração falhou para os parâmetros "
                                 "calibrados (penalidade aplicada). "
                                 "Resultados anteriores preservados.")
                    else:
                        st.session_state.sim_result = res
                        st.session_state.sim_params = {
                            "Rc": float(p[0]), "m": float(p[1]),
                            "theta0_deg": float(np.degrees(p[2])),
                            "delta_theta_deg": float(np.degrees(p[3])),
                            "method": opts.get("method", "DOP853"),
                            "rtol": opts.get("rtol", 1e-9),
                            "atol": opts.get("atol", 1e-9),
                        }
                        st.success("Simulação executada com os parâmetros "
                                   "calibrados — veja a aba **Results**.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Falha: {e}")

        st.markdown("#### Salvar calibração")
        st.download_button(
            "⬇ Salvar calibração (JSON)",
            data=rep.calibration_json_bytes(
                cal,
                engine_params=_engine_params_dict(
                    st.session_state.engine_cfg),
                data_name=st.session_state.data_name,
                meta=st.session_state.calib_meta,
                theta=st.session_state.data_theta,
                pressure=st.session_state.data_press),
            file_name="calibracao_single_wiebe.json",
            mime="application/json",
            help="Salva parâmetros calibrados, histórico da busca, tabela "
                 "e os dados experimentais usados — reabra no botão "
                 "“📂 Carregar calibração”, junto ao Run calibration.")


# =============================================================================
# Aba 5 — Results
# =============================================================================
with tabs[4]:
    st.subheader("Results")
    res = st.session_state.sim_result
    cal = st.session_state.calib_result

    r1, r2 = st.columns(2)
    with r1:
        angle_unit = st.radio("Unidade angular", ["rad", "graus"], 0,
                              horizontal=True)
    with r2:
        p_unit = st.radio("Unidade de pressão", ["kPa", "bar"], 0,
                          horizontal=True)

    if res is None:
        st.info("Execute uma simulação (aba **Simulation**) para ver os "
                "resultados.")
    else:
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(pl.fig_pressure(res, angle_unit, p_unit),
                            use_container_width=True)
            st.plotly_chart(pl.fig_burned(res, angle_unit),
                            use_container_width=True)
            st.plotly_chart(pl.fig_temperature(res, angle_unit),
                            use_container_width=True)
            st.plotly_chart(pl.fig_volume(res, angle_unit,
                                          st.session_state.engine_cfg),
                            use_container_width=True)
        with p2:
            st.plotly_chart(pl.fig_residual(res, angle_unit, p_unit),
                            use_container_width=True)
            st.plotly_chart(pl.fig_heat_release(res, angle_unit),
                            use_container_width=True)
            st.plotly_chart(pl.fig_heat_loss(res, angle_unit),
                            use_container_width=True)
            st.plotly_chart(pl.fig_pv(res), use_container_width=True)

        hist_fonte = cal if (cal is not None and cal.get("history")) else None
        st.plotly_chart(pl.fig_convergence(
            hist_fonte["history"] if hist_fonte else [],
            method_name=(f"melhor erro {hist_fonte['erro']:.4f} kPa "
                         f"({hist_fonte['iteracoes']} iterações)"
                         if hist_fonte else "execute a calibração na aba 4")),
            use_container_width=True)

        st.markdown("#### Tabela de resíduos")
        df_res = rep.results_dataframe(res, st.session_state.engine_cfg)
        df_tab = df_res[["theta_rad", "pressure_exp_kPa", "pressure_sim_kPa"]].copy()
        df_tab["residuo_kPa"] = df_res["residual_kPa"]
        df_tab["residuo_%"] = (100.0 * df_tab["residuo_kPa"]
                               / df_tab["pressure_exp_kPa"].replace(0, np.nan))
        st.dataframe(df_tab.round(4), use_container_width=True, hide_index=True)


# =============================================================================
# Aba 6 — Export
# =============================================================================
with tabs[5]:
    st.subheader("Export")
    res = st.session_state.sim_result
    if res is None or res.P_sim is None:
        st.info("Nenhum resultado válido para exportar — execute uma "
                "simulação bem-sucedida na aba **Simulation**.")
    else:
        cfg = st.session_state.engine_cfg
        metrics = {
            "erro_notebook_kPa": res.erro,
            "RMSE_kPa": sw.rmse(res.P_exp, res.P_sim),
            "R2": sw.r_squared(res.P_exp, res.P_sim),
            "P_max_kPa": float(res.P_sim.max()),
            "T_max_K": float(res.Tg_sim.max()),
            "Qp_final_J": float(res.Qp_sim[-1]),
        }
        engine_params = _engine_params_dict(cfg)
        cal = st.session_state.calib_result
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("⬇ Resultados completos (CSV)",
                               data=rep.results_csv_bytes(res, cfg),
                               file_name="resultados_single_wiebe.csv",
                               mime="text/csv")
            st.download_button("⬇ Parâmetros (JSON)",
                               data=rep.params_json_bytes(
                                   res, cfg, engine_params, metrics,
                                   st.session_state.data_name),
                               file_name="parametros_single_wiebe.json",
                               mime="application/json")
            hist_ok = cal is not None and bool(cal.get("history"))
            st.download_button("⬇ Histórico da otimização (CSV)",
                               data=(rep.history_csv_bytes(
                                   cal["history"], cal["history_params"],
                                   param_names=cal.get("selected"))
                                   if hist_ok else b""),
                               file_name="historico_otimizacao.csv",
                               mime="text/csv",
                               disabled=not hist_ok)
        with e2:
            figs = pl.export_static_plots(res)
            import io as _io
            png_buf = _io.BytesIO()
            figs["pressao_exp_sim"].savefig(png_buf, format="png", dpi=200,
                                            bbox_inches="tight")
            st.download_button("⬇ Gráfico principal (PNG)", data=png_buf.getvalue(),
                               file_name="pressao_exp_sim.png",
                               mime="image/png")

            pdf_buf = _io.BytesIO()
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(pdf_buf) as pdf:
                for fig in figs.values():
                    pdf.savefig(fig, bbox_inches="tight")
            st.download_button("⬇ Todos os gráficos (PDF)", data=pdf_buf.getvalue(),
                               file_name="graficos_single_wiebe.pdf",
                               mime="application/pdf")

            cal = st.session_state.calib_result
            figs_png = {}
            import io as _io2
            for nome, fig in figs.items():
                b = _io2.BytesIO()
                fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
                figs_png[nome] = b.getvalue()
            rel = rep.report_html_bytes(
                res, cfg, engine_params, metrics,
                calibration_table=(cal["tabela"] if cal is not None else None),
                alertas=(cal["alertas"] if cal is not None else None),
                data_name=st.session_state.data_name,
                figures_png=figs_png,
            )
            st.download_button("⬇ Relatório (HTML)", data=rel,
                               file_name="relatorio_single_wiebe.html",
                               mime="text/html")

        st.caption("O CSV segue o formato: theta_rad; theta_deg; "
                   "pressure_exp_kPa; pressure_sim_kPa; residual_kPa; "
                   "temperature_K; heat_loss_J; burned_fraction; "
                   "dQ_dtheta_kJ_per_rad; volume_m3 (separador ';').")