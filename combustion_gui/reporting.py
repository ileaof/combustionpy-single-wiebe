# -*- coding: utf-8 -*-
"""
reporting.py
===============================================================================
Exportações da GUI Single Wiebe: CSV completo, JSON de parâmetros,
relatório resumido em HTML e histórico da otimização. Nenhuma equação aqui —
somente formatação dos resultados produzidos por single_wiebe.py.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import single_wiebe as sw

GUI_CSV_COLUMNS = [
    "theta_rad",
    "theta_deg",
    "pressure_exp_kPa",
    "pressure_sim_kPa",
    "residual_kPa",
    "temperature_K",
    "heat_loss_J",
    "burned_fraction",
    "dQ_dtheta_kJ_per_rad",
    "volume_m3",
]


# ---------------------------------------------------------------------------
# CSV completo (10 colunas, separador ';', vírgula decimal não — usar '.')
# ---------------------------------------------------------------------------
def results_dataframe(res: sw.ModelResult,
                      cfg: Optional[sw.EngineConfig] = None) -> pd.DataFrame:
    """Monta o DataFrame de exportação com as 10 colunas da GUI."""
    V = sw.cylinder_volume(res.theta, res.Rc, cfg)
    return pd.DataFrame({
        "theta_rad": res.theta,
        "theta_deg": np.degrees(res.theta),
        "pressure_exp_kPa": res.P_exp,
        "pressure_sim_kPa": res.P_sim,
        "residual_kPa": res.P_exp - res.P_sim,
        "temperature_K": res.Tg_sim,
        "heat_loss_J": res.Qp_sim,
        "burned_fraction": res.x_b,
        "dQ_dtheta_kJ_per_rad": res.dQ_dtheta,
        "volume_m3": V,
    })


def results_csv_bytes(res: sw.ModelResult,
                      cfg: Optional[sw.EngineConfig] = None) -> bytes:
    df = results_dataframe(res, cfg)
    return df.to_csv(index=False, sep=";", float_format="%.9f").encode("utf-8")


# ---------------------------------------------------------------------------
# JSON de parâmetros
# ---------------------------------------------------------------------------
def params_json_bytes(res: sw.ModelResult, cfg: sw.EngineConfig,
                      engine_params: Dict, metrics: Dict,
                      data_name: str = "") -> bytes:
    """Serializa identificação, motor, combustão, método e métricas."""
    doc = {
        "analise": {
            "titulo": "Single Wiebe Combustion Analysis",
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "arquivo_experimental": data_name,
        },
        "motor": engine_params | {
            "area_pistao_m2": cfg.A_cyl,
            "volume_deslocado_m3": cfg.Vd,
            "velocidade_media_pistao_m_s": cfg.Vp,
            "rotacao_rev_s": cfg.omega_rev_s,
            "razao_biela_manivela": cfg.R_rod_ratio,
        },
        "combustao": {
            "Rc": res.Rc,
            "m": res.m,
            "theta0_deg": float(np.degrees(res.theta0)),
            "delta_theta_deg": float(np.degrees(res.delta_theta)),
            "a_wiebe": cfg.a_wiebe,
            "transferencia_de_calor": cfg.heat_transfer,
        },
        "metodo_numerico": {
            "integrador": "DOP853 (solve_ivp)",
            "rtol": sw.RTOL_DEFAULT,
            "atol": sw.ATOL_DEFAULT,
            "t_eval": "ângulos experimentais (sem interpolação)",
        },
        "metricas_ajuste": metrics,
    }
    return json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Arquivo de calibração (salvar / abrir)
# ---------------------------------------------------------------------------
_CALIB_FORMATO = "single-wiebe-calibracao"
_CALIB_VERSAO = 1


def calibration_json_bytes(cal: Dict,
                           engine_params: Optional[Dict] = None,
                           data_name: str = "",
                           meta: Optional[Dict] = None,
                           theta: Optional[np.ndarray] = None,
                           pressure: Optional[np.ndarray] = None) -> bytes:
    """Serializa o resultado da calibração (dicionário `calib_result` da GUI)
    num arquivo JSON reaberto por `read_calibration_json`.

    Unidades: ângulos em radianos (formato interno — o mesmo vetor que
    `simulate_full` consome). `theta`/`pressure` embutem os dados
    experimentais usados na calibração; `meta` registra como a busca rodou
    (método, semente, população...) e `engine_params` o motor, só para
    rastreabilidade.
    """
    doc = {
        "analise": {
            "titulo": "Single Wiebe Combustion Analysis — Calibração",
            "formato": _CALIB_FORMATO,
            "versao": _CALIB_VERSAO,
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "arquivo_experimental": data_name,
        },
        "motor": engine_params or {},
        "busca": meta or {},
        "dados": {
            "theta_rad": ([float(v) for v in theta]
                          if theta is not None else []),
            "pressao_kPa": ([float(v) for v in pressure]
                            if pressure is not None else []),
        },
        "calibracao": {
            "selected": list(cal.get("selected", [])),
            "Rc": float(cal["params"][0]),
            "m": float(cal["params"][1]),
            "theta0_rad": float(cal["params"][2]),
            "delta_theta_rad": float(cal["params"][3]),
            "erro_kPa": float(cal["erro"]),
            "iteracoes": int(cal.get("iteracoes", 0)),
            "cancelado": bool(cal.get("cancelado", False)),
            "parou_por_repeticao": bool(cal.get("parou_por_repeticao", False)),
            "polish": cal.get("polish"),
        },
        "tabela": cal.get("tabela", []),
        "alertas": list(cal.get("alertas", [])),
        "historico": {
            "erro_kPa": [float(h) for h in cal.get("history", [])],
            "parametros": [[float(v) for v in p]
                           for p in cal.get("history_params", [])],
        },
    }
    return json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")


def read_calibration_json(arquivo) -> Dict:
    """Lê um arquivo salvo por `calibration_json_bytes` e devolve um
    dicionário com: 'calibracao' (o `calib_result` reconstruído, ângulos em
    rad), 'theta'/'pressao' (dados embutidos como par, ou None), 'motor',
    'busca' e 'arquivo_experimental'. Lança ValueError em arquivo inválido.
    """
    try:
        texto = arquivo.read() if hasattr(arquivo, "read") else str(arquivo)
        if isinstance(texto, bytes):
            texto = texto.decode("utf-8")
        doc = json.loads(texto)
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"não é um JSON válido ({e}).") from e
    if (not isinstance(doc, dict)
            or (doc.get("analise") or {}).get("formato") != _CALIB_FORMATO):
        raise ValueError('esperado um arquivo salvo pela GUI '
                         f'("formato": "{_CALIB_FORMATO}").')
    c = doc.get("calibracao") or {}
    if any(k not in c for k in ("Rc", "m", "theta0_rad", "delta_theta_rad",
                                "erro_kPa")):
        raise ValueError('seção "calibracao" incompleta '
                         "(Rc/m/theta0_rad/delta_theta_rad/erro_kPa).")
    params = np.array([float(c["Rc"]), float(c["m"]),
                       float(c["theta0_rad"]), float(c["delta_theta_rad"])])
    if not np.all(np.isfinite(params)):
        raise ValueError("parâmetros não finitos no arquivo.")

    hist = doc.get("historico") or {}
    calib = {
        "params": params,
        "selected": list(c.get("selected", [])),
        "erro": float(c["erro_kPa"]),
        "iteracoes": int(c.get("iteracoes", 0)),
        "history": [float(h) for h in (hist.get("erro_kPa") or [])],
        "history_params": [np.asarray(p, dtype=float)
                           for p in (hist.get("parametros") or [])],
        "parou_por_repeticao": bool(c.get("parou_por_repeticao", False)),
        "cancelado": bool(c.get("cancelado", False)),
        "polish": c.get("polish"),
        "tabela": list(doc.get("tabela") or []),
        "alertas": list(doc.get("alertas") or []),
    }

    dados = doc.get("dados") or {}
    theta = (np.asarray(dados["theta_rad"], dtype=float)
             if dados.get("theta_rad") else None)
    pressao = (np.asarray(dados["pressao_kPa"], dtype=float)
               if dados.get("pressao_kPa") else None)
    if theta is None or pressao is None or theta.size != pressao.size:
        theta = pressao = None       # só restaura o PAR completo

    return {
        "calibracao": calib,
        "theta": theta,
        "pressao": pressao,
        "motor": dict(doc.get("motor") or {}),
        "busca": dict(doc.get("busca") or {}),
        "arquivo_experimental": str((doc.get("analise") or {})
                                    .get("arquivo_experimental") or ""),
    }


# ---------------------------------------------------------------------------
# Histórico da otimização
# ---------------------------------------------------------------------------
_OPT_PARAM_NAMES = ["Rc", "m", "theta0", "delta_theta"]


def history_csv_bytes(history: List[float],
                      history_params: List[np.ndarray],
                      param_names: Optional[List[str]] = None) -> bytes:
    """Histórico da otimização: erro e parâmetros por iteração.

    `param_names` nomeia as colunas dos parâmetros (default: nomes canônicos
    na ordem do vetor — quando apenas parte dos parâmetros foi calibrada,
    a GUI passa os nomes do subvetor).
    """
    df = pd.DataFrame({
        "iteracao": np.arange(1, len(history) + 1),
        "erro_kPa": history,
    })
    names = param_names or _OPT_PARAM_NAMES
    for i, nome in enumerate(names):
        unidade = "deg" if nome in ("theta0", "delta_theta") else "-"
        df[f"{nome}_{unidade}"] = [p[i] if i < len(p) else np.nan
                                   for p in history_params]
    return df.to_csv(index=False, sep=";", float_format="%.9g").encode("utf-8")


# ---------------------------------------------------------------------------
# Relatório resumido em HTML
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Relatório — Single Wiebe Combustion Analysis</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
          margin: 0; background: #f6f7f9; color: #1d2430; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #dfe3ea; padding-bottom: 8px; }}
  h2 {{ font-size: 17px; margin-top: 26px; color: #0b62d6; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 8px 0; }}
  th, td {{ border: 1px solid #dfe3ea; padding: 6px 10px; text-align: left; }}
  th {{ background: #e8f1fd; }}
  .meta {{ color: #5b6472; font-size: 13px; }}
  .warn {{ background: #fff4d6; border: 1px solid #d9a62e; border-radius: 6px;
           padding: 8px 12px; margin: 8px 0; font-size: 14px; }}
  img {{ max-width: 100%; border: 1px solid #dfe3ea; border-radius: 8px;
         margin: 10px 0; }}
</style>
</head>
<body><main>
<h1>{titulo}</h1>
<p class="meta">Data/hora: {data_hora}<br>
Arquivo experimental: {arquivo}</p>

<h2>Parâmetros do motor</h2>
{tabela_motor}

<h2>Parâmetros de combustão</h2>
{tabela_combustao}

<h2>Método numérico</h2>
<p>Integrador: {integrador} — rtol/atol = {tol}<br>
Avaliação exata nos ângulos experimentais (t_eval, sem interpolação).</p>

<h2>Métricas de ajuste</h2>
{tabela_metricas}

<h2>Parâmetros calibrados</h2>
{tabela_calibrados}

<h2>Alertas e observações</h2>
{alertas}

<h2>Gráficos principais</h2>
{graficos}
</main></body></html>
"""


def report_html_bytes(res: sw.ModelResult, cfg: sw.EngineConfig,
                      engine_params: Dict, metrics: Dict,
                      calibration_table: Optional[List[Dict]] = None,
                      alertas: Optional[List[str]] = None,
                      data_name: str = "",
                      figures_png: Optional[Dict[str, bytes]] = None) -> bytes:
    """Gera o relatório HTML autocontido (imagens embutidas em base64)."""
    def _tabela(linhas: List[Dict]) -> str:
        if not linhas:
            return "<p><i>não disponível</i></p>"
        cols = list(linhas[0].keys())
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join(
            "<tr>" + "".join((f"<td>{lin[c]:.6g}</td>" if isinstance(lin[c], float)
                              else f"<td>{lin[c]}</td>") for c in cols) + "</tr>"
            for lin in linhas
        )
        return (f"<table><tr>{head}</tr>{body}</table>")

    motor_rows = [{"Parâmetro": k, "Valor": v} for k, v in engine_params.items()]
    comb_rows = [
        {"Parâmetro": "Rc [-]", "Valor": res.Rc},
        {"Parâmetro": "m [-]", "Valor": res.m},
        {"Parâmetro": "θ₀ [°]", "Valor": float(np.degrees(res.theta0))},
        {"Parâmetro": "Δθ [°]", "Valor": float(np.degrees(res.delta_theta))},
    ]
    metricas_rows = [{"Métrica": k, "Valor": f"{v:.6g}" if isinstance(v, float)
                      else str(v)} for k, v in metrics.items()]

    calib_rows = calibration_table or []

    alertas_html = ("<ul>" + "".join(f"<li class='warn'>{a}</li>"
                                     for a in (alertas or [])) + "</ul>"
                    ) if alertas else "<p><i>nenhum alerta.</i></p>"

    graficos_html = ""
    if figures_png:
        import base64
        for nome, png in figures_png.items():
            b64str = base64.b64encode(png).decode("ascii")
            graficos_html += f"<img src='data:image/png;base64,{b64str}' alt='{nome}'>"

    html = _HTML_TEMPLATE.format(
        titulo="Single Wiebe Combustion Analysis — Relatório",
        data_hora=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        arquivo=data_name or "(não informado)",
        tabela_motor=_tabela(motor_rows),
        tabela_combustao=_tabela(comb_rows),
        integrador="DOP853 (scipy solve_ivp)",
        tol=f"{sw.RTOL_DEFAULT:g}",
        tabela_metricas=_tabela(metricas_rows),
        tabela_calibrados=_tabela(calib_rows),
        alertas=alertas_html,
        graficos=graficos_html or "<p><i>nenhum gráfico incluído.</i></p>",
    )
    return html.encode("utf-8")