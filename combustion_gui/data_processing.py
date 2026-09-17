# -*- coding: utf-8 -*-
"""
data_processing.py
===============================================================================
Camada de processamento de dados da GUI Single Wiebe.

Responsabilidades (sem lógica física — as equações ficam em single_wiebe.py):
  * leitura flexível de arquivos .txt/.csv/.tsv (separador, colunas, unidades);
  * conversões de unidades (ângulo -> rad; pressão -> kPa);
  * filtro de intervalo angular, ordenação e suavização opcional;
  * validações numéricas (NaN/inf, dimensões, ordenação);
  * resumo estatístico do conjunto carregado.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Unidades aceitas e fatores de conversão para kPa
PRESSURE_FACTORS_KPA = {"bar": 100.0, "kPa": 1.0, "Pa": 0.001}
ANGLE_FACTORS_RAD = {"radianos": 1.0, "graus": np.pi / 180.0}


class DataError(Exception):
    """Erro de leitura/validação com mensagem amigável para a interface."""


# ---------------------------------------------------------------------------
# Leitura flexível
# ---------------------------------------------------------------------------
def read_table(path_or_buffer, sep: str = "auto") -> pd.DataFrame:
    """Lê o arquivo experimental em um DataFrame numérico.

    `sep` aceita qualquer caractere ou "auto" (csv.Sniffer com fallback
    para whitespace, que cobre tabulações e espaços como no arquivo típico).
    """
    try:
        if sep == "auto":
            try:
                # Lê uma amostra para detectar o separador
                if hasattr(path_or_buffer, "seek"):
                    path_or_buffer.seek(0)
                sample = path_or_buffer.read(4096)
                if hasattr(path_or_buffer, "seek"):
                    path_or_buffer.seek(0)
                dialect_sep = None
                if isinstance(sample, bytes):
                    sample = sample.decode("utf-8", errors="replace")
                import csv
                try:
                    dialect_sep = csv.Sniffer().sniff(sample).delimiter
                except csv.Error:
                    dialect_sep = None
                # whitespace puro (" ") confunde o sniffer: usa pandas p/ tabs
                if dialect_sep in (None, "", " "):
                    sep_used = r"\s+"
                else:
                    sep_used = dialect_sep
            except Exception:
                sep_used = r"\s+"
        else:
            sep_used = r"\s+" if sep == "whitespace" else sep

        df = pd.read_csv(
            path_or_buffer,
            sep=sep_used,
            header=None,
            engine="python",
            comment="#",
            dtype=str,
        )
        # Converte tudo para numérico; células não numéricas viram NaN e são
        # descartadas se remove_invalid=True (ver prepare_series).
        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace(",", ".",
                                                            regex=False),
                errors="coerce",
            )
        # Descarta colunas totalmente vazias (ex.: separador duplicado)
        df = df.dropna(axis=1, how="all")
        if df.shape[1] < 2:
            raise DataError(
                "Não foi possível identificar 2 colunas numéricas "
                "(ângulo e pressão). Verifique o separador."
            )
        return df
    except pd.errors.EmptyDataError as e:
        raise DataError("O arquivo está vazio.") from e
    except UnicodeDecodeError as e:
        raise DataError("Não foi possível decodificar o arquivo como texto "
                        "(UTF-8/ASCII).") from e


def prepare_series(
    df: pd.DataFrame,
    angle_col: int,
    pressure_col: int,
    angle_unit: str = "radianos",
    pressure_unit: str = "bar",
    theta_min: Optional[float] = None,
    theta_max: Optional[float] = None,
    remove_invalid: bool = True,
    sort_by_angle: bool = True,
    smooth: bool = False,
    smooth_window: int = 11,
    smooth_order: int = 2,
    min_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Converte, filtra e valida as séries angulares de pressão.

    Retorna (theta_rad, pressure_kPa, resumo). Levanta DataError com
    mensagem clara quando algo impede a análise.
    """
    if angle_col >= df.shape[1] or pressure_col >= df.shape[1]:
        raise DataError(
            f"Coluna selecionada fora do arquivo (o arquivo tem "
            f"{df.shape[1]} colunas)."
        )

    theta = df.iloc[:, angle_col].to_numpy(dtype=float)
    P = df.iloc[:, pressure_col].to_numpy(dtype=float)

    if remove_invalid:
        valid = np.isfinite(theta) & np.isfinite(P)
        n_dropped = int((~valid).sum())
        theta, P = theta[valid], P[valid]
    else:
        n_dropped = 0
        if not (np.isfinite(theta).all() and np.isfinite(P).all()):
            raise DataError(
                "O arquivo contém linhas não numéricas (NaN/inf). "
                "Ative 'Remover linhas inválidas'."
            )

    if theta.size == 0:
        raise DataError("Nenhuma observação numérica válida no arquivo.")

    # Conversões de unidade (para rad e kPa — unidades internas do modelo)
    if angle_unit not in ANGLE_FACTORS_RAD:
        raise DataError(f"Unidade angular desconhecida: {angle_unit}")
    if pressure_unit not in PRESSURE_FACTORS_KPA:
        raise DataError(f"Unidade de pressão desconhecida: {pressure_unit}")
    theta = theta * ANGLE_FACTORS_RAD[angle_unit]
    P = P * PRESSURE_FACTORS_KPA[pressure_unit]

    # Filtro de intervalo angular (em rad, após conversão)
    mask = np.ones_like(theta, dtype=bool)
    if theta_min is not None:
        mask &= theta >= float(theta_min)
    if theta_max is not None:
        mask &= theta <= float(theta_max)
    theta, P = theta[mask], P[mask]
    if theta.size < min_points:
        raise DataError(
            f"Somente {theta.size} observações no intervalo angular "
            f"selecionado — mínimo exigido: {min_points}."
        )

    if sort_by_angle:
        order = np.argsort(theta)
        theta, P = theta[order], P[order]
    elif np.any(np.diff(theta) < 0):
        raise DataError(
            "Ângulos não estão ordenados. Ative 'Ordenar por ângulo'."
        )

    if smooth:
        window = min(smooth_window, (P.size // 2) * 2 - 1)
        if window >= smooth_order + 2 and window >= 3:
            P = savgol_filter(P, window, smooth_order)

    # Validações finais
    if not (np.isfinite(theta).all() and np.isfinite(P).all()):
        raise DataError("Séries contêm NaN/inf após o processamento.")
    if np.any(P <= 0):
        raise DataError("Pressões não positivas no conjunto (P <= 0).")

    resumo = {
        "n_obs": int(theta.size),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "P_min": float(P.min()),
        "P_max": float(P.max()),
        "passo_medio": float(np.mean(np.diff(np.sort(theta)))) if theta.size > 1 else 0.0,
        "n_descartadas": n_dropped,
    }
    return theta, P, resumo


def load_file_full(path_or_buffer, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Atalho: read_table + prepare_series com validação de erro amigável."""
    df = read_table(path_or_buffer, sep=kwargs.pop("sep", "auto"))
    return prepare_series(df, **kwargs)