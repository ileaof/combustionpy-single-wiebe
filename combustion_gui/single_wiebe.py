# -*- coding: utf-8 -*-
"""
single_wiebe.py
===============================================================================
Conversão para Python (NumPy/SciPy) do modelo de combustão "Single Wiebe"
originalmente implementado em Wolfram Mathematica
(Modelo_Single_Wiebe_v1.nb, funções MOD0d e programa PSO).

MODO DE COMPATIBILIDADE
-----------------------
A função :func:`simulate_model` reproduz, 1-para-1, a lógica do notebook:

  * geometria biela-manivela V(theta), dV/dtheta, y(theta), As(theta);
  * fração queimada Single Wiebe x_b(theta) e sua derivada;
  * liberação de calor Q = m_comb * PCI * x_b  [kJ] e dQ/dtheta [kJ/rad];
  * correlação de Hohenberg para o coeficiente de película h;
  * sistema de 3 EDOs em theta:  [P (kPa), Tg (K), Qp (J/rad acumulado)];
  * erro = sqrt(SSres / (q - 2))  -- ver observação em :func:`compute_error`.

Melhorias (físicas/numéricas) ficam claramente sinalizadas na seção
"MELHORIAS" no fim deste arquivo, separadas do modo de compatibilidade.

UNIDADES
--------
  theta       : rad (todos os cálculos são em radianos, como no notebook)
  P           : kPa   (dados experimentais entram em bar e são * 100)
  Tg, T1, Tw  : K
  V, Vd, Vc   : m³
  Q, dQ/dtheta: kJ, kJ/rad
  Qp          : J (energia perdida acumulada por radiano -> total em J)
  h           : W/(m²·K)
  omega       : rotações por SEGUNDO (rpm/60), como no notebook

Autoria: conversão fiel do notebook original.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# =============================================================================
# Constantes do motor (idênticas às do notebook)
# =============================================================================
D_BORE = 86.0 / 1000.0        # d  - diâmetro do cilindro [m]
S_STROKE = 70.0 / 1000.0      # s  - curso do pistão [m]
L_ROD = 117.5 / 1000.0        # l  - comprimento da biela [m]
N_CYL = 1                     # número de cilindros
KAPPA = 1.37                  # kp - razão de capacidades caloríficas [-]
RPM = 3396.20                 # rotação [rpm]
OMEGA_REV_S = RPM / 60.0      # ω  - rotação por SEGUNDO [rev/s]
R_CRANK = S_STROKE / 2.0      # r  - raio da manivela [m]
R_ROD_RATIO = L_ROD / R_CRANK # R  - razão biela/manivela (l/r) [-]
A_CYL = np.pi * (D_BORE / 2.0) ** 2   # área do pistão [m²]
VD = A_CYL * S_STROKE         # volume deslocado [m³]
VP = 2.0 * S_STROKE * OMEGA_REV_S     # velocidade média do pistão [m/s]

M_COMB = 9.42754647351e-6     # massa de combustível por ciclo [kg/ciclo]
PCI = 39191.3                 # poder calorífico inferior [kJ/kg]
A_WIEBE = 6.9078              # parâmetro de eficiência da combustão (a) [-]
TW = 440.0                    # temperatura da parede [K]
T1 = 273.15 + 35.0            # temperatura no IVC (início da integração) [K]

# Tolerâncias padrão do integrador (ver README: diferenças Mathematica x SciPy)
RTOL_DEFAULT = 1e-9
ATOL_DEFAULT = 1e-9

# Penalidade retornada quando a integração falha ou gera valores não físicos
PENALTY = 1.0e10


# =============================================================================
# 0. Configuração do motor [EXTENSÃO GUI]
# =============================================================================
# Permite que a interface gráfica altere geometria, rotação e propriedades
# sem duplicar as equações: todas as funções físicas aceitam `cfg=None`,
# caso em que usam exatamente as constantes do modo de compatibilidade.
@dataclass
class EngineConfig:
    """Parâmetros físicos do motor usados pelas funções do modelo.

    Os defaults reproduzem exatamente as constantes do notebook
    (modo de compatibilidade); a GUI cria uma instância editada pelo usuário.
    """
    bore: float = D_BORE            # diâmetro do cilindro [m]
    stroke: float = S_STROKE        # curso do pistão [m]
    rod_length: float = L_ROD       # comprimento da biela [m]
    rpm: float = RPM                # rotação [rpm]
    kappa: float = KAPPA            # razão de capacidades caloríficas [-]
    m_comb: float = M_COMB          # massa de combustível [kg/ciclo]
    pci: float = PCI                # poder calorífico inferior [kJ/kg]
    a_wiebe: float = A_WIEBE        # parâmetro de eficiência do Wiebe [-]
    T1: float = T1                  # temperatura no IVC [K]
    Tw: float = TW                  # temperatura da parede [K]
    heat_transfer: bool = True      # ativa/desativa a perda de calor (Qp)

    @property
    def r_crank(self) -> float:
        """Raio da manivela r = s/2 [m]."""
        return self.stroke / 2.0

    @property
    def R_rod_ratio(self) -> float:
        """Razão biela-manivela R = l/r [-]."""
        return self.rod_length / self.r_crank

    @property
    def A_cyl(self) -> float:
        """Área de seção transversal do cilindro [m²]."""
        return np.pi * (self.bore / 2.0) ** 2

    @property
    def Vd(self) -> float:
        """Volume deslocado Vd = A_cil * s [m³]."""
        return self.A_cyl * self.stroke

    @property
    def omega_rev_s(self) -> float:
        """Rotação em revoluções por segundo [rev/s]."""
        return self.rpm / 60.0

    @property
    def Vp(self) -> float:
        """Velocidade média do pistão Vp = 2*s*omega [m/s]."""
        return 2.0 * self.stroke * self.omega_rev_s

    def validate(self) -> List[str]:
        """Retorna lista de mensagens de erro para valores não físicos."""
        erros = []
        for nome, valor, vmin in (
            ("diâmetro", self.bore, 1e-4),
            ("curso", self.stroke, 1e-4),
            ("comprimento da biela", self.rod_length, 1e-4),
            ("rotação", self.rpm, 1.0),
            ("massa de combustível", self.m_comb, 0.0),
            ("PCI", self.pci, 1.0),
            ("a (Wiebe)", self.a_wiebe, 0.0),
        ):
            if not (valor > vmin):
                erros.append(f"{nome} deve ser > {vmin} (recebido {valor}).")
        if not (1.0 < self.kappa < 2.0):
            erros.append(f"kappa deve estar em (1, 2) (recebido {self.kappa}).")
        if self.T1 <= 0 or self.Tw <= 0:
            erros.append("Temperaturas (T1, Tw) devem ser positivas [K].")
        if self.rod_length <= self.r_crank:
            erros.append("Biela deve ser mais longa que o raio da manivela "
                         "(l > s/2) para a geometria ser válida.")
        return erros


DEFAULT_CONFIG = EngineConfig()


# =============================================================================
# 1. Leitura e preparação dos dados experimentais
# =============================================================================
def load_experimental(path: str) -> Dict[str, np.ndarray]:
    """Lê o arquivo de pressão experimental (theta [rad], P [bar]).

    - separação por espaços ou tabulações (whitespace);
    - descarta linhas inválidas/vazias;
    - filtra -2 <= theta <= 2 rad;
    - converte pressão de bar para kPa (x 100);
    - ordena por ângulo crescente.
    """
    raw = np.loadtxt(path, comments="#")  # whitspace/tab: loadtxt cobre ambos
    raw = raw[np.isfinite(raw).all(axis=1)]

    mask = (raw[:, 0] >= -2.0) & (raw[:, 0] <= 2.0)
    theta = np.sort(raw[mask, 0])         # ordenação por ângulo (crescente)
    pressure_kPa = 100.0 * raw[mask, 1][np.argsort(raw[mask, 0])]

    # Conjunto bruto completo (equivalente ao tablePall do notebook), para o
    # gráfico GrafPall da primeira figura do PDF. Em kPa (o notebook plotava
    # os valores em bar com rótulo kPa — inconsistência corrigida aqui).
    theta_raw = np.sort(raw[:, 0])
    pressure_raw_kPa = 100.0 * raw[:, 1][np.argsort(raw[:, 0])]

    data = {
        "theta": theta,           # rad (filtrado)
        "pressure": pressure_kPa, # kPa (filtrado)
        "theta_raw": theta_raw,           # rad (503 pontos, sem filtro)
        "pressure_raw": pressure_raw_kPa, # kPa (503 pontos, sem filtro)
        "n_total": raw.shape[0],
        "n_filtered": theta.size,
        "P1": pressure_kPa[0],    # pressão no IVC (primeiro ponto filtrado)
        "theta_i": theta[0],
        "theta_f": theta[-1],
    }
    if raw.shape[0] != 503:
        raise ValueError(
            f"Esperadas 503 observações no arquivo completo; obtidas {raw.shape[0]}."
        )
    if theta.size != 459:
        raise ValueError(
            f"Esperadas 459 observações após o filtro; obtidas {theta.size}."
        )
    return data


# =============================================================================
# 2. Geometria biela-manivela
# =============================================================================
def piston_disp(theta, Rc: float, cfg: Optional[EngineConfig] = None) -> np.ndarray:
    """y(theta): deslocamento do pistão a partir do PMS [m].

    y = l + r - r*cos(theta) - sqrt(l² - r² sin²(theta))
    """
    c = cfg or DEFAULT_CONFIG
    l, r = c.rod_length, c.r_crank
    return l + r - r * np.cos(theta) - np.sqrt(l**2 - r**2 * np.sin(theta) ** 2)


def cylinder_volume(theta, Rc: float, cfg: Optional[EngineConfig] = None) -> np.ndarray:
    """V(theta): volume instantâneo do cilindro [m³].

    V = Vd/(Rc-1) + (Vd/2) * [R + 1 - cos(theta) - sqrt(R² - sin²(theta))]
    """
    c = cfg or DEFAULT_CONFIG
    Vd, R = c.Vd, c.R_rod_ratio
    return Vd / (Rc - 1.0) + (Vd / 2.0) * (
        R
        + 1.0
        - np.cos(theta)
        - np.sqrt(R**2 - np.sin(theta) ** 2)
    )


def dV_dtheta(theta, Rc: float, cfg: Optional[EngineConfig] = None) -> np.ndarray:
    """dV/dtheta [m³/rad]."""
    c = cfg or DEFAULT_CONFIG
    Vd, R = c.Vd, c.R_rod_ratio
    return (Vd * np.sin(theta) / 2.0) * (
        1.0 + np.cos(theta) / np.sqrt(R**2 - np.sin(theta) ** 2)
    )


def heat_area(theta, Rc: float, cfg: Optional[EngineConfig] = None) -> np.ndarray:
    """As(theta): área de transferência de calor [m²].

    As = 2*pi*(d/2)² + pi*d*y(theta) + pi*d*s/(Rc-1)
    """
    c = cfg or DEFAULT_CONFIG
    return (
        2.0 * np.pi * (c.bore / 2.0) ** 2
        + np.pi * c.bore * piston_disp(theta, Rc, c)
        + np.pi * c.bore * c.stroke / (Rc - 1.0)
    )


# =============================================================================
# 3. Função de fração queimada Single Wiebe
# =============================================================================
def burned_fraction(theta, theta0: float, delta_theta: float, m: float,
                    cfg: Optional[EngineConfig] = None):
    """x_b(theta): fração queimada (Single Wiebe) e sua derivada dx/dtheta.

    Para theta >= theta0:
        z = (theta - theta0)/delta_theta
        x_b = 1 - exp(-a * z^(m+1))
        dx_b/dtheta = a*(m+1)/delta_theta * z^m * exp(-a * z^(m+1))
    Para theta < theta0: x_b = 0 e dx_b/dtheta = 0.

    MODO DE COMPATIBILIDADE: a expressão NÃO é truncada em
    theta0 + delta_theta, pois o Mathematica continua avaliando a expressão
    além desse ponto (x_b -> 1 assintoticamente, dx_b/dtheta -> 0).

    Retorna (x_b, dx_b_dtheta) com o mesmo shape de theta.
    """
    c = cfg or DEFAULT_CONFIG
    a = c.a_wiebe
    theta = np.asarray(theta, dtype=float)
    z = (theta - theta0) / delta_theta
    active = theta >= theta0
    # z^m e z^(m+1) calculados com z>=0 para evitar NaN; fora do domínio z<0
    # os resultados são zerados pela máscara `active`. Com m<0 (não físico)
    # 0**m = inf — caso tratado pelo mecanismo de penalidade da simulação.
    z_safe = np.where(active, np.maximum(z, 0.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zn = z_safe ** m                    # z^m  (0**0 = 1, como em Mathematica)
        znp1 = z_safe ** (m + 1.0)          # z^(m+1)
        expo = np.exp(-a * znp1)
        x = np.where(active, 1.0 - expo, 0.0)
        dx = np.where(active, a * (m + 1.0) / delta_theta * zn * expo, 0.0)
    return x, dx


def heat_release(theta, theta0: float, delta_theta: float, m: float,
                 cfg: Optional[EngineConfig] = None):
    """Calor liberado acumulado Q [kJ] e derivada dQ/dtheta [kJ/rad].

    Q(theta)      = m_comb * PCI * x_b(theta)
    dQ/dtheta     = m_comb * PCI * dx_b/dtheta
    """
    c = cfg or DEFAULT_CONFIG
    x, dx = burned_fraction(theta, theta0, delta_theta, m, c)
    return c.m_comb * c.pci * x, c.m_comb * c.pci * dx


# =============================================================================
# 4. Correlação de Hohenberg
# =============================================================================
def hohenberg_h(theta, P_kPa, Tg, Rc: float, cfg: Optional[EngineConfig] = None):
    """Coeficiente de transferência de calor por convecção [W/(m²·K)].

    h = 130 * V^-0.06 * (P*1e-2)^0.8 * Tg^-0.4 * (Vp + 1.4)^0.8

    P em kPa (o código original multiplica por 10^-2 apenas para converter
    kPa -> bar, escala usada na correlação de Hohenberg).
    """
    c = cfg or DEFAULT_CONFIG
    V = cylinder_volume(theta, Rc, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (
            130.0
            * V ** (-0.06)
            * (P_kPa * 1.0e-2) ** 0.8
            * Tg ** (-0.4)
            * (c.Vp + 1.4) ** 0.8
        )


# =============================================================================
# 5. Sistema de EDOs  Y = [P (kPa), Tg (K), Qp (J)]
# =============================================================================
class _ModelBlowUp(Exception):
    """[interno] Sinaliza RHS não física (usada para penalizar rápido a
    função objetivo em vez de deixar o integrador afundar em passos ~0)."""


def _make_rhs(Rc: float, theta0: float, delta_theta: float, m: float,
              P1: float, V1: float,
              cfg: Optional[EngineConfig] = None) -> Callable:
    """Constrói o lado direito do sistema de EDOs (função pura, sem globals).

    dP/dtheta  = 1/V * [ (kappa-1)*(dQ/dtheta - dQp/1000) - kappa*P*dV/dtheta ]
    dTg/dtheta = T1/(P1*V1) * [ V*dP/dtheta + P*dV/dtheta ]
    dQp/dtheta = h(theta,P,Tg) * As(theta) * (Tg - Tw) / (2*pi*omega)

    dQ/dtheta em kJ/rad; dQp em J/rad (dividir por 1000 -> kJ/rad no balanço);
    h*As*(Tg-Tw) em W; dividir por 2*pi*omega [rad/s] converte em J/rad.
    """
    c = cfg or DEFAULT_CONFIG
    kappa, T1, Tw = c.kappa, c.T1, c.Tw
    # dV/dtheta pré-calculado é função de theta apenas; deixado inline abaixo.
    _LIMIT = 1.0e12  # limite de sanidade para as derivadas (valores >= são não físicos)

    def rhs(theta: float, y: np.ndarray) -> np.ndarray:
        P, Tg, _ = y
        V = cylinder_volume(theta, Rc, c)
        dV = dV_dtheta(theta, Rc, c)
        _, dQ = heat_release(theta, theta0, delta_theta, m, c)   # kJ/rad
        As = heat_area(theta, Rc, c)
        if c.heat_transfer:
            h = hohenberg_h(theta, P, Tg, Rc, c)                 # W/(m² K)
            dQp = h * As * (Tg - Tw) / (2.0 * np.pi * c.omega_rev_s)  # J/rad
        else:
            dQp = 0.0                                            # sem perda

        dP = (1.0 / V) * (
            (kappa - 1.0) * (dQ - dQp / 1000.0) - kappa * P * dV
        )                                                     # kPa/rad
        dTg = (T1 / (P1 * V1)) * (V * dP + P * dV)            # K/rad
        out = np.array([dP, dTg, dQp])
        if not np.all(np.isfinite(out)) or np.max(np.abs(out)) > _LIMIT:
            raise _ModelBlowUp
        return out

    return rhs


def simulate_model(
    Rc: float,
    m: float,
    theta0: float,
    delta_theta: float,
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    method: str = "DOP853",
    cfg: Optional[EngineConfig] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve o sistema de EDOs nos ângulos experimentais e devolve o erro.

    Assinatura pedida:
        simulate_model(Rc, m, theta0, delta_theta, theta_exp, pressure_exp)
    -> (erro, P_sim, Tg_sim, Qp_sim)

    - integra de theta_i = theta_exp[0] até theta_f = theta_exp[-1];
    - P1 = pressure_exp[0] (kPa), T1 = 308.15 K, Qp = 0 (IVC);
    - a solução é avaliada exatamente nos ângulos experimentais via
      ``t_eval`` (sem interpolação externa);
    - se a integração falhar ou gerar valores não físicos (NaN, P<=0, T<=0),
      retorna a penalidade :data:`PENALTY` sem interromper a otimização.
    """
    c = cfg or DEFAULT_CONFIG
    theta_i = float(theta_exp[0])
    theta_f = float(theta_exp[-1])
    P1 = float(pressure_exp[0])
    V1 = float(cylinder_volume(theta_i, Rc, c))

    rhs = _make_rhs(Rc, theta0, delta_theta, m, P1, V1, c)
    y0 = np.array([P1, c.T1, 0.0])

    try:
        sol = solve_ivp(
            rhs,
            (theta_i, theta_f),
            y0,
            method=method,
            t_eval=theta_exp,
            rtol=rtol,
            atol=atol,
        )
    except (ValueError, FloatingPointError, OverflowError, _ModelBlowUp):
        return PENALTY, None, None, None

    if not sol.success or sol.y.shape[1] != theta_exp.size or not np.all(
        np.isfinite(sol.y)
    ):
        return PENALTY, None, None, None

    P_sim, Tg_sim, Qp_sim = sol.y

    # Verificação de fisicalidade (pressões/temperaturas positivas)
    if np.any(P_sim <= 0.0) or np.any(Tg_sim <= 0.0):
        return PENALTY, None, None, None

    erro = compute_error(pressure_exp, P_sim)
    return erro, P_sim, Tg_sim, Qp_sim


def compute_error(P_exp: np.ndarray, P_sim: np.ndarray) -> float:
    """erro = sqrt( SSres / (q - 2) ).

    PARTICULARIDADE PRESERVADA DO NOTEBOOK: o divisor é (q - 2) e não
    (q - n_parâmetros), embora quatro parâmetros sejam ajustados. Essa é a
    definição exata usada no Mathematica (erro = Sqrt[SSres/(q - 2.)]) e é
    mantida para reproduzir os valores de referência (409.385 kPa na
    validação; 63.8473 kPa na calibração). Ver README.
    """
    q = P_exp.size
    SSres = np.sum((P_exp - P_sim) ** 2)
    return float(np.sqrt(SSres / (q - 2.0)))


def rmse(P_exp: np.ndarray, P_sim: np.ndarray) -> float:
    """Raiz do erro quadrático médio: sqrt( mean( (Pexp - Psim)^2 ) ) [kPa]."""
    return float(np.sqrt(np.mean((np.asarray(P_exp) - np.asarray(P_sim)) ** 2)))


def r_squared(P_exp: np.ndarray, P_sim: np.ndarray) -> float:
    """Coeficiente de determinação R² = 1 - SSres/SStot [-]."""
    P_exp = np.asarray(P_exp, dtype=float)
    P_sim = np.asarray(P_sim, dtype=float)
    SStot = float(np.sum((P_exp - P_exp.mean()) ** 2))
    if SStot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((P_exp - P_sim) ** 2)) / SStot


# =============================================================================
# 6. Calibração
# =============================================================================
# Limites do domínio (idênticos ao notebook):
#   Rc          : 15 a 17
#   m           : 0.1 a 1.0
#   theta0      : -1° a 2°      (em rad internamente)
#   delta_theta : 40° a 60°     (em rad internamente)
LOWER = np.array([15.0, 0.1, np.radians(-1.0), np.radians(40.0)])
UPPER = np.array([17.0, 1.0, np.radians(2.0), np.radians(60.0)])

PSO_DEFAULTS = dict(
    n_particulas=20,
    beta=2.0,
    max_iteracoes=800,
    max_repeticoes=10,
    tolerancia=1e-10,
)


def calibrate_pso(
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    n_particulas: int = 20,
    beta: float = 2.0,
    max_iteracoes: int = 800,
    max_repeticoes: int = 10,
    tolerancia: float = 1e-10,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """PSO compatível com o notebook (implementação definitiva).

    Réplica do While[kr < krmax && k < kmax] do Mathematica:
      k  : contador de iterações;
      kr : iterações consecutivas em que a melhor partícula global mudou
           menos que `tolerancia` (max-norma); zerado quando ultrapassa.
    """
    rng = np.random.default_rng(seed)
    nvar = 4
    li, ls = LOWER, UPPER

    def fitness(x: np.ndarray) -> float:
        return simulate_model(*x, theta_exp, pressure_exp)[0]

    # População inicial
    X = li + rng.random((n_particulas, nvar)) * (ls - li)
    fX = np.array([fitness(X[j]) for j in range(n_particulas)])

    P_best = X.copy()
    f_best = fX.copy()
    jbest = int(np.argmin(f_best))
    fbest = float(f_best[jbest])

    kr = 0
    xv_old = P_best[jbest].copy()
    history: List[float] = []
    history_params: List[np.ndarray] = []

    while kr < max_repeticoes and len(history) < max_iteracoes:
        r1 = rng.random((n_particulas, nvar))
        r2 = rng.random((n_particulas, nvar))
        X = X + beta * r1 * (P_best - X) + beta * r2 * (P_best[jbest] - X)
        X = np.clip(X, li, ls)

        fX = np.array([fitness(X[j]) for j in range(n_particulas)])

        improved = fX < f_best
        P_best[improved] = X[improved]
        f_best[improved] = fX[improved]

        jnew = int(np.argmin(f_best))
        if f_best[jnew] < fbest:
            fbest = float(f_best[jnew])
            jbest = jnew

        xv = P_best[jbest]
        ermax = float(np.max(np.abs(xv - xv_old)))
        xv_old = xv.copy()

        kr = kr + 1 if ermax <= tolerancia else 0
        history.append(fbest)
        history_params.append(P_best[jbest].copy())
        if verbose:
            print(f"PSO it {len(history):4d}: erro = {fbest:.6f} kPa | "
                  f"params = {np.round(P_best[jbest], 6)}")

    best = P_best[jbest].copy()
    return {
        "params": best,
        "Rc": best[0],
        "m": best[1],
        "theta0_rad": best[2],
        "theta0_deg": np.degrees(best[2]),
        "delta_theta_rad": best[3],
        "delta_theta_deg": np.degrees(best[3]),
        "erro": fbest,
        "iteracoes": len(history),
        "history": history,
        "history_params": history_params,
        "parou_por_repeticao": kr >= max_repeticoes,
    }


def calibrate_de(
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    seed: Optional[int] = None,
    maxiter: int = 300,
    popsize: int = 20,
    tol: float = 1e-10,
    polish: bool = False,
) -> Dict:
    """MÉTODO ALTERNATIVO (robusto): scipy.optimize.differential_evolution.

    Não faz parte do modo de compatibilidade; serve como verificação
    independente do ótimo do PSO simplificado do notebook.
    """
    bounds = list(zip(LOWER, UPPER))

    def objective(x: np.ndarray) -> float:
        return simulate_model(*x, theta_exp, pressure_exp)[0]

    de_history: List[float] = []

    def callback(xk, convergence=0):
        # Registra o melhor fitness da população a cada iteração do DE.
        de_history.append(objective(xk))
        return False

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        updating="immediate",
        callback=callback,
    )

    best = result.x.copy()
    return {
        "params": best,
        "Rc": best[0],
        "m": best[1],
        "theta0_rad": best[2],
        "theta0_deg": np.degrees(best[2]),
        "delta_theta_rad": best[3],
        "delta_theta_deg": np.degrees(best[3]),
        "erro": float(result.fun),
        "iteracoes": int(result.nit),
        "history": de_history,
        "success": bool(result.success),
        "message": str(result.message),
    }


# =============================================================================
# 6b. Calibração generalizada [EXTENSÃO GUI]
# =============================================================================
# Mesmos algoritmos do modo de compatibilidade, com limites arbitrários,
# parâmetros fixos, configuração do motor, retorno de progresso e
# cancelamento cooperativo (para a GUI rodar a otimização em thread).
def calibrate_pso_bounds(
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: Optional[int] = None,
    n_particulas: int = 20,
    beta: float = 2.0,
    max_iteracoes: int = 800,
    max_repeticoes: int = 10,
    tolerancia: float = 1e-10,
    cfg: Optional[EngineConfig] = None,
    progress_callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    expand: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict:
    """PSO do notebook com limites/parametrização arbitrários.

    `progress_callback(iteracao, erro, params)` é chamada a cada iteração;
    `cancel_check()` avaliada ao fim de cada iteração — se retornar True,
    o laço para e o resultado traz "cancelado": True.
    `expand(x)` mapeia o vetor otimizado para o vetor completo de parâmetros
    (usado quando apenas parte dos parâmetros é calibrada; default: identidade).
    """
    rng = np.random.default_rng(seed)
    li = np.asarray(lower, dtype=float)
    ls = np.asarray(upper, dtype=float)
    nvar = li.size

    def fitness(x: np.ndarray) -> float:
        xf = expand(x) if expand is not None else x
        return simulate_model(*xf, theta_exp, pressure_exp, cfg=cfg)[0]

    X = li + rng.random((n_particulas, nvar)) * (ls - li)
    fX = np.array([fitness(X[j]) for j in range(n_particulas)])

    P_best = X.copy()
    f_best = fX.copy()
    jbest = int(np.argmin(f_best))
    fbest = float(f_best[jbest])

    kr = 0
    xv_old = P_best[jbest].copy()
    history: List[float] = []
    history_params: List[np.ndarray] = []
    cancelado = False

    while kr < max_repeticoes and len(history) < max_iteracoes:
        r1 = rng.random((n_particulas, nvar))
        r2 = rng.random((n_particulas, nvar))
        X = X + beta * r1 * (P_best - X) + beta * r2 * (P_best[jbest] - X)
        X = np.clip(X, li, ls)

        fX = np.array([fitness(X[j]) for j in range(n_particulas)])

        improved = fX < f_best
        P_best[improved] = X[improved]
        f_best[improved] = fX[improved]

        jnew = int(np.argmin(f_best))
        if f_best[jnew] < fbest:
            fbest = float(f_best[jnew])
            jbest = jnew

        xv = P_best[jbest]
        ermax = float(np.max(np.abs(xv - xv_old)))
        xv_old = xv.copy()

        kr = kr + 1 if ermax <= tolerancia else 0
        history.append(fbest)
        history_params.append(P_best[jbest].copy())
        if progress_callback is not None:
            progress_callback(len(history), fbest, P_best[jbest].copy())
        if cancel_check is not None and cancel_check():
            cancelado = True
            break

    best = P_best[jbest].copy()
    return {
        "params": best,
        "erro": fbest,
        "iteracoes": len(history),
        "history": history,
        "history_params": history_params,
        "parou_por_repeticao": kr >= max_repeticoes,
        "cancelado": cancelado,
    }


def calibrate_de_bounds(
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: Optional[int] = None,
    maxiter: int = 300,
    popsize: int = 20,
    tol: float = 1e-10,
    polish: bool = False,
    cfg: Optional[EngineConfig] = None,
    progress_callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    expand: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict:
    """Differential evolution com limites arbitrários + progresso/cancelamento.

    Retornar True de `cancel_check` interrompe o DE (scipy trata callback
    True como parada). `expand(x)` mapeia o vetor otimizado para o vetor
    completo de parâmetros (default: identidade).
    """
    bounds = list(zip(np.asarray(lower, float), np.asarray(upper, float)))

    def objective(x: np.ndarray) -> float:
        xf = expand(x) if expand is not None else x
        return simulate_model(*xf, theta_exp, pressure_exp, cfg=cfg)[0]

    history: List[float] = []
    history_params: List[np.ndarray] = []

    def callback(intermediate_result=None, **_):
        # scipy >= 1.9 passa IntermediateResult; manter compatível com xk.
        x = (intermediate_result.x
             if intermediate_result is not None and hasattr(intermediate_result, "x")
             else _.get("xk"))
        f = (float(intermediate_result.fun)
             if intermediate_result is not None and hasattr(intermediate_result, "fun")
             else objective(x))
        history.append(f)
        history_params.append(np.asarray(x, dtype=float).copy())
        if progress_callback is not None:
            progress_callback(len(history), f, history_params[-1])
        if cancel_check is not None and cancel_check():
            return True  # solicita parada do DE
        return False

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        updating="immediate",
        callback=callback,
    )

    best = result.x.copy()
    return {
        "params": best,
        "erro": float(result.fun),
        "iteracoes": int(result.nit),
        "history": history,
        "history_params": history_params,
        "success": bool(result.success),
        "message": str(result.message),
        "cancelado": bool(not result.success and "stopped" in result.message.lower()),
    }


# =============================================================================
# 7. MELHORIAS (fora do modo de compatibilidade)
# =============================================================================
# Melhoria 1 (física): x_b deveria ser truncado em theta0 + delta_theta
# (combustão completa); o notebook deixa a cauda assintótica do Wiebe ativa.
def wiebe_truncated(theta, theta0, delta_theta, m):
    """[MELHORIA] Wiebe com combustão encerrada em theta0 + delta_theta."""
    theta = np.asarray(theta, dtype=float)
    x, dx = burned_fraction(theta, theta0, delta_theta, m)
    beyond = theta > theta0 + delta_theta
    x = np.where(beyond, 1.0, x)
    dx = np.where(beyond, 0.0, dx)
    return x, dx


# Melhoria 2 (numérica): integrar em "tempo" adimensional (theta/theta_f - 1)
# para equiparar escalas; na prática o intervalo é curto e DOP853 já é
# adequado, então o ganho é marginal. Registrada aqui apenas como nota.
# Melhoria 3 (numérica): acelerar o PSO avaliando partículas em paralelo
# (multiprocessing) — não altera resultados, só o custo computacional.


@dataclass
class ModelResult:
    """Agrupa os resultados de uma simulação completa para plotagem/export."""
    Rc: float
    m: float
    theta0: float
    delta_theta: float
    erro: float
    theta: np.ndarray
    P_exp: np.ndarray
    P_sim: np.ndarray
    Tg_sim: np.ndarray
    Qp_sim: np.ndarray
    x_b: np.ndarray
    dQ_dtheta: np.ndarray
    extras: Dict = field(default_factory=dict)


# =============================================================================
# 8. Simulação completa (para gráficos e exportação)
# =============================================================================
def simulate_full(
    Rc: float,
    m: float,
    theta0: float,
    delta_theta: float,
    theta_exp: np.ndarray,
    pressure_exp: np.ndarray,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    method: str = "DOP853",
    cfg: Optional[EngineConfig] = None,
) -> ModelResult:
    """Roda :func:`simulate_model` e completa com x_b e dQ/dtheta."""
    erro, P_sim, Tg_sim, Qp_sim = simulate_model(
        Rc, m, theta0, delta_theta, theta_exp, pressure_exp, rtol, atol,
        method=method, cfg=cfg,
    )
    x, dx = burned_fraction(theta_exp, theta0, delta_theta, m, cfg)
    return ModelResult(
        Rc=Rc, m=m, theta0=theta0, delta_theta=delta_theta, erro=erro,
        theta=theta_exp, P_exp=pressure_exp, P_sim=P_sim, Tg_sim=Tg_sim,
        Qp_sim=Qp_sim, x_b=x, dQ_dtheta=dx,
    )


# =============================================================================
# 9. Gráficos
# =============================================================================
def make_plots(res: ModelResult, calibration: Optional[Dict] = None,
               outdir: str = "results",
               data: Optional[Dict] = None) -> List[str]:
    """Gera e salva os gráficos do estudo em `outdir`.

    As quatro primeiras figuras replicam os gráficos do notebook/PDF
    (`GrafPall`, `GrafPbar`, `GrafPExp` e `Show[GrafPExp, GrafPSim]`),
    no mesmo estilo: pontos experimentais como marcadores "+" vermelhos e
    curva simulada como linha verde contínua.

    ATENÇÃO à rotulagem: todos os eixos de ângulo estão em RADIANOS (os
    cálculos são feitos em radianos; algumas figuras originais do notebook
    mencionavam graus por engano — p. ex. o rótulo "dQdθ [kJ/deg]", embora
    a grandeza seja kJ/rad).

    `data` (opcional) é o dicionário retornado por `load_experimental`;
    quando fornecido, inclui a figura dos dados brutos completos (GrafPall).
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files: List[str] = []
    th = res.theta
    xlab = "Ângulo do Eixo de Manivelas [rad]"

    def _salvar(fig, nome):
        p = os.path.join(outdir, nome)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # ------------------------------------------------------------------
    # 1. GrafPall — TODOS os pontos brutos do arquivo (503), kPa x rad.
    #    (No notebook a curva era plotada em bar com rótulo "kPa";
    #    aqui os valores são convertidos e o rótulo é coerente.)
    # ------------------------------------------------------------------
    if data is not None and "theta_raw" in data:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(data["theta_raw"], data["pressure_raw"], "r+", ms=4)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Pressão no Interior do Cilindro [kPa]")
        ax.set_title("Dados experimentais completos (GrafPall)")
        _salvar(fig, "01_pressao_exp_bruta_kpa_rad.png")

    # ------------------------------------------------------------------
    # 2. GrafPbar — dados filtrados (-2 <= theta <= 2), em bar x rad.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.P_exp / 100.0, "r+", ms=4)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Pressão no Interior do Cilindro [bar]")
    ax.set_title("Pressão experimental filtrada -2 a 2 rad (GrafPbar)")
    _salvar(fig, "02_pressao_exp_filtrada_bar_rad.png")

    # ------------------------------------------------------------------
    # 3. GrafPExp — dados filtrados em kPa x rad, marcadores "+" vermelhos.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.P_exp, "r+", ms=4)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Pressão no Interior do Cilindro [kPa]")
    ax.set_title("Pressão experimental filtrada (GrafPExp)")
    _salvar(fig, "03_pressao_exp_filtrada_kpa_rad.png")

    # ------------------------------------------------------------------
    # 4. Show[GrafPExp, GrafPSim] — experimental ("+") + simulada (verde).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(th, res.P_exp, "r+", ms=4, label="Experimental")
    ax.plot(th, res.P_sim, "g-", lw=1.4, label="Simulada (Single Wiebe)")
    ax.set_xlabel("Rad")
    ax.set_ylabel("Pressão [kPa]")
    ax.set_title(
        f"Pressão simulada x experimental  |  erro = {res.erro:.3f} kPa"
    )
    ax.legend(loc="upper right")
    _salvar(fig, "04_pressao_exp_sim_kpa_rad.png")

    # ------------------------------------------------------------------
    # 5. Fração queimada x_b
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.x_b, "k-", lw=1.2)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Fração queimada x_b [-]")
    ax.set_title("Fração queimada (Single Wiebe)")
    ax.grid(True)
    _salvar(fig, "05_fracao_queimada.png")

    # ------------------------------------------------------------------
    # 6. dQ/dtheta — linha verde, como no GrafdQdθ do notebook
    #    (rótulo original dizia "kJ/deg"; a grandeza é kJ/rad).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.dQ_dtheta, "g-", lw=1.4)
    ax.set_xlabel(xlab)
    ax.set_ylabel("dQdθ [kJ/rad]")
    ax.set_title("Taxa de liberação de calor (GrafdQdθ)")
    _salvar(fig, "06_dQ_dtheta_kJ_rad.png")

    # ------------------------------------------------------------------
    # 7. Temperatura do gás
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.Tg_sim, "m-", lw=1.2)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Temperatura do gás [K]")
    ax.set_title("Temperatura do gás")
    ax.grid(True)
    _salvar(fig, "07_temperatura_gas.png")

    # ------------------------------------------------------------------
    # 8. Calor perdido acumulado (Qp em J)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.Qp_sim, "c-", lw=1.2)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Calor perdido acumulado [J]")
    ax.set_title("Calor perdido para as paredes (Hohenberg)")
    ax.grid(True)
    _salvar(fig, "08_calor_perdido.png")

    # ------------------------------------------------------------------
    # 9. Convergência da função objetivo (PSO e/ou DE)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if calibration is not None:
        hist = calibration.get("history") or []
        if hist:
            ax.plot(range(1, len(hist) + 1), hist, "b-o", ms=3,
                    label=f"{'PSO' if 'parou_por_repeticao' in calibration else 'DE'}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("erro = sqrt(SSres/(q-2)) [kPa]")
    ax.set_title("Convergência da função objetivo")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _salvar(fig, "09_convergencia.png")

    return files


# =============================================================================
# 10. Exportação
# =============================================================================
CSV_COLUMNS = [
    "theta_rad",
    "theta_deg",
    "pressure_exp_kPa",
    "pressure_sim_kPa",
    "temperature_K",
    "heat_loss_J",
    "burned_fraction",
    "dQ_dtheta_kJ_per_rad",
]


def export_csv(res: ModelResult, outdir: str = "results") -> str:
    """Exporta a tabela principal (ângulos experimentais) em CSV."""
    import csv
    import os
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "resultados_simulacao.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(CSV_COLUMNS)
        for i in range(res.theta.size):
            writer.writerow([
                f"{res.theta[i]:.9f}",
                f"{np.degrees(res.theta[i]):.6f}",
                f"{res.P_exp[i]:.6f}",
                f"{res.P_sim[i]:.6f}",
                f"{res.Tg_sim[i]:.6f}",
                f"{res.Qp_sim[i]:.6f}",
                f"{res.x_b[i]:.9f}",
                f"{res.dQ_dtheta[i]:.9f}",
            ])
    return path


def export_pressure_degrees_bar(
    Rc: float, m: float, theta0: float, delta_theta: float,
    theta_i: float, theta_f: float, P1: float,
    outdir: str = "results",
    step_deg: float = 0.34,
) -> str:
    """Tabela de pressão simulada em graus e bar com passo ~0.34°.

    Equivalente à exportação original do notebook: refaz a integração com
    t_eval na grade de graus (sem interpolação). `P1` é a pressão no IVC
    [kPa], a mesma condição inicial do modo de compatibilidade.
    """
    import csv
    import os
    os.makedirs(outdir, exist_ok=True)

    # Grade uniforme em graus, convertida para radianos
    theta_grid = np.arange(theta_i, theta_f + 1e-12, np.radians(step_deg))
    if theta_grid[-1] < theta_f:
        theta_grid = np.append(theta_grid, theta_f)

    V1 = float(cylinder_volume(theta_i, Rc))
    rhs = _make_rhs(Rc, theta0, delta_theta, m, P1, V1)
    sol = solve_ivp(
        rhs, (theta_i, theta_f), np.array([P1, T1, 0.0]),
        method="DOP853", t_eval=theta_grid,
        rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT,
    )
    if not sol.success:
        raise RuntimeError(f"Integração da grade em graus falhou: {sol.message}")
    P_sim_kPa = sol.y[0]

    path = os.path.join(outdir, "pressao_simulada_graus_bar.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["theta_deg", "pressure_sim_bar"])
        for i in range(theta_grid.size):
            writer.writerow([
                f"{np.degrees(theta_grid[i]):.6f}",
                f"{P_sim_kPa[i] / 100.0:.6f}",
            ])
    return path