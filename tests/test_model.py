# -*- coding: utf-8 -*-
"""
test_model.py — testes do modelo Single Wiebe.

Executar a partir da pasta do pacote:
    python -m pytest tests/ -v
ou diretamente:
    python tests/test_model.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import single_wiebe as sw  # noqa: E402

# Dados: pasta pai do pacote (layout original, arquivo fora do repositório)
# ou raiz do repositório (clone autocontido, cópia incluída no git).
_candidates = [
    Path(__file__).resolve().parents[2] / "P_exp-Carga-3_45%.txt",
    Path(__file__).resolve().parents[1] / "P_exp-Carga-3_45%.txt",
]
DATA_PATH = next(p for p in _candidates if p.exists())

# Caso de validação do notebook
VALID_PARAMS = (17.0, 0.504, math.radians(-6.54), math.radians(62.0))
VALID_ERRO_REF = 409.385


# ---------------------------------------------------------------------------
# Dados experimentais
# ---------------------------------------------------------------------------
def test_leitura_503_observacoes():
    raw = np.loadtxt(DATA_PATH)
    assert raw.shape == (503, 2)


def test_filtro_459_pontos():
    data = sw.load_experimental(str(DATA_PATH))
    assert data["n_filtered"] == 459
    th = data["theta"]
    assert np.all(th >= -2.0) and np.all(th <= 2.0)
    assert np.all(np.diff(th) > 0)  # ordenado crescente
    passo = np.median(np.diff(th))
    assert abs(passo - 0.008726646) < 1e-6  # ~0.5 deg


# ---------------------------------------------------------------------------
# Geometria biela-manivela
# ---------------------------------------------------------------------------
def test_volume_no_pms():
    # V(theta=0) = Vc = Vd/(Rc-1)
    Rc = 17.0
    assert abs(sw.cylinder_volume(0.0, Rc) - sw.VD / (Rc - 1.0)) < 1e-15


def test_volume_maximo_no_pmi():
    # V(theta=pi) deve ser praticamente Vc + Vd
    Rc = 17.0
    Vpmi = sw.cylinder_volume(np.pi, Rc)
    assert abs(Vpmi - (sw.VD / (Rc - 1.0) + sw.VD)) < 1e-8 * Vpmi


def test_deslocamento_pistao_nos_extremos():
    # y(0) = 0 (PMS) e y(pi) ~ 2r = s (PMI)
    assert abs(sw.piston_disp(0.0, 17.0)) < 1e-15
    assert abs(sw.piston_disp(np.pi, 17.0) - 2 * sw.R_CRANK) < 1e-12


def test_dv_dtheta_numerico():
    # derivada analítica vs diferença finita central
    Rc = 16.0
    th = 0.7
    h = 1e-7
    num = (sw.cylinder_volume(th + h, Rc) - sw.cylinder_volume(th - h, Rc)) / (2 * h)
    ana = sw.dV_dtheta(th, Rc)
    assert abs(num - ana) < 1e-9 * abs(ana) + 1e-12


def test_area_transferencia_em_theta_zero():
    Rc = 17.0
    As0 = sw.heat_area(0.0, Rc)
    esperado = 2 * np.pi * (sw.D_BORE / 2) ** 2 + 0.0 + np.pi * sw.D_BORE * sw.S_STROKE / (Rc - 1.0)
    assert abs(As0 - esperado) < 1e-15


# ---------------------------------------------------------------------------
# Função de Wiebe
# ---------------------------------------------------------------------------
def test_wiebe_zero_antes_do_inicio():
    x, dx = sw.burned_fraction(np.array([-1.0, -0.5, -0.1]), theta0=0.0,
                               delta_theta=0.5, m=0.5)
    assert np.all(x == 0.0)
    assert np.all(dx == 0.0)


def test_wiebe_inicia_em_zero():
    x, _ = sw.burned_fraction(0.0, theta0=0.0, delta_theta=0.5, m=0.5)
    assert abs(x) < 1e-15


def test_wiebe_derivada_vs_numerica():
    m, th0, dth = 0.504, 0.1, 1.0
    th = th0 + 0.4
    h = 1e-7
    num = (sw.burned_fraction(th + h, th0, dth, m)[0]
           - sw.burned_fraction(th - h, th0, dth, m)[0]) / (2 * h)
    ana = sw.burned_fraction(th, th0, dth, m)[1]
    assert abs(num - ana) < 1e-6


def test_wiebe_modo_compatibilidade_nao_trunca():
    # No modo de compatibilidade a fração continua evoluindo além de
    # theta0 + delta_theta (sem corte em 1.0 antes do ponto avaliado):
    # em z>1 a derivada é positiva (x ainda cresce), nunca fica constante.
    th0, dth, m = 0.0, 1.0, 0.5
    z_teste = th0 + dth + 0.2
    x1, _ = sw.burned_fraction(z_teste, th0, dth, m)
    x2, _ = sw.burned_fraction(z_teste + 0.1, th0, dth, m)
    assert x2 > x1  # ainda crescente fora do intervalo


# ---------------------------------------------------------------------------
# Sistema de EDOs / simulação
# ---------------------------------------------------------------------------
def test_simulacao_dimensoes_e_sem_nan():
    data = sw.load_experimental(str(DATA_PATH))
    erro, P, Tg, Qp = sw.simulate_model(*VALID_PARAMS,
                                        data["theta"], data["pressure"])
    n = data["theta"].size
    assert P.shape == (n,) and Tg.shape == (n,) and Qp.shape == (n,)
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(Tg)) and np.all(np.isfinite(Qp))


def test_condicoes_iniciais():
    data = sw.load_experimental(str(DATA_PATH))
    erro, P, Tg, Qp = sw.simulate_model(*VALID_PARAMS,
                                        data["theta"], data["pressure"])
    assert abs(P[0] - data["P1"]) < 1e-6
    assert abs(Tg[0] - sw.T1) < 1e-9
    assert abs(Qp[0]) < 1e-9


def test_validacao_caso_notebook():
    """Caso obrigatório: erro ~ 409.385 kPa (tolerância 0.1 kPa)."""
    data = sw.load_experimental(str(DATA_PATH))
    erro, *_ = sw.simulate_model(*VALID_PARAMS, data["theta"], data["pressure"])
    assert abs(erro - VALID_ERRO_REF) < 0.1, f"erro={erro}, ref={VALID_ERRO_REF}"


def test_penalidade_parametros_ruins():
    data = sw.load_experimental(str(DATA_PATH))
    # Caso genuinamente não físico: expoente de Wiebe negativo com theta0
    # coincidindo exatamente com o início da integração (z = 0 na condição
    # inicial -> derivada explode); a integração deve falhar e a função
    # objetivo devolver penalidade alta, não exceção.
    theta0 = data["theta"][0]
    erro, P, Tg, Qp = sw.simulate_model(
        16.0, -2.0, theta0, 1.0, data["theta"], data["pressure"])
    assert erro >= sw.PENALTY and P is None


if __name__ == "__main__":
    import sys as _sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} testes passaram.")
    _sys.exit(0)