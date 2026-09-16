# -*- coding: utf-8 -*-
"""
run_model.py
===============================================================================
Script de execução do modelo Single Wiebe (conversão do notebook Mathematica).

Fluxo:
  1. leitura e preparação dos dados experimentais;
  2. validação obrigatória do caso de referência (erro ~ 409.385 kPa);
  3. calibração por PSO (modo de compatibilidade com o Mathematica) e,
     como método alternativo robusto, differential_evolution (SciPy);
  4. gráficos (results/*.png) e exportação dos CSVs (results/*.csv).

Uso:
    python run_model.py                       # fluxo completo (seed 42)
    python run_model.py --seed 123            # outra semente do PSO
    python run_model.py --skip-de             # só validação + PSO
    python run_model.py --data "..\\P_exp-Carga-3_45%.txt"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import single_wiebe as sw

# ---------------------------------------------------------------------------
# Caminho padrão dos dados experimentais: pasta pai do pacote (layout
# original) ou a própria pasta do pacote (clone autocontido do repositório).
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_data_name = "P_exp-Carga-3_45%.txt"
_default_candidates = [
    os.path.join(os.path.dirname(_this_dir), _data_name),
    os.path.join(_this_dir, _data_name),
]
DEFAULT_DATA = next(p for p in _default_candidates if os.path.exists(p))

# Caso de validação obrigatória (valores do notebook)
VALID_RC = 17.0
VALID_M = 0.504
VALID_THETA0_DEG = -6.54
VALID_DELTA_THETA_DEG = 62.0
VALID_ERRO_REF = 409.385  # kPa, conforme o notebook Mathematica


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Modelo Single Wiebe - conversão Python do notebook Mathematica"
    )
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="caminho do arquivo P_exp (theta [rad], P [bar])")
    parser.add_argument("--seed", type=int, default=42,
                        help="semente aleatória do PSO/DE (default: 42)")
    parser.add_argument("--iters", type=int, default=None,
                        help="sobrescreve max_iteracoes do PSO (padrão: 800)")
    parser.add_argument("--skip-pso", action="store_true",
                        help="não rodar a calibração PSO")
    parser.add_argument("--skip-de", action="store_true",
                        help="não rodar a calibração differential_evolution")
    parser.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"),
        help="diretório de saída (gráficos e CSVs)")
    parser.add_argument("--quiet", action="store_true",
                        help="suprime o log por iteração do PSO")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("MODELO SINGLE WIEBE - conversao Python do notebook Mathematica")
    print("=" * 74)

    # ------------------------------------------------------------------ 1. Dados
    data = sw.load_experimental(args.data)
    theta_exp = data["theta"]
    P_exp = data["pressure"]
    q = theta_exp.size
    print(f"\n[1] Dados experimentais: {data['n_total']} linhas lidas, "
          f"{q} após o filtro -2 <= theta <= 2 rad")
    print(f"    theta_i = {theta_exp[0]:.9f} rad ({np.degrees(theta_exp[0]):.4f} deg)")
    print(f"    theta_f = {theta_exp[-1]:.9f} rad ({np.degrees(theta_exp[-1]):.4f} deg)")
    print(f"    passo   = {np.median(np.diff(theta_exp)):.9f} rad (0.5 deg)")
    print(f"    P1 (IVC) = {data['P1']:.2f} kPa | T1 = {sw.T1} K")

    # ------------------------------------------------------------- 2. Validação
    print("\n[2] Validacao do caso de referencia do notebook")
    print(f"    Rc = {VALID_RC}, m = {VALID_M}, theta0 = {VALID_THETA0_DEG} deg, "
          f"delta_theta = {VALID_DELTA_THETA_DEG} deg")
    t0 = time.perf_counter()
    erro_val, Ps, Tg, Qp = sw.simulate_model(
        VALID_RC, VALID_M,
        np.radians(VALID_THETA0_DEG), np.radians(VALID_DELTA_THETA_DEG),
        theta_exp, P_exp,
    )
    dt = time.perf_counter() - t0
    diff = erro_val - VALID_ERRO_REF
    print(f"    erro Python  = {erro_val:.6f} kPa")
    print(f"    erro notebook = {VALID_ERRO_REF} kPa")
    print(f"    diferenca     = {diff:+.6f} kPa ({100*diff/VALID_ERRO_REF:+.5f}%) "
          f"[1 avaliacao em {dt*1000:.0f} ms]")
    print(f"    P_sim max = {Ps.max():.1f} kPa | Tg max = {Tg.max():.1f} K | "
          f"Qp final = {Qp[-1]:.2f} J")

    if abs(diff) > 1.0:
        print("    AVISO: diferenca acima de 1 kPa - investigar unidades, "
              "tolerancias do integrador e ordem das equacoes.")

    best = None
    label = None

    # ------------------------------------------------------- 3a. Calibração PSO
    if not args.skip_pso:
        print(f"\n[3a] Calibracao PSO (compativel com o notebook, seed={args.seed})")
        print("     limites: Rc [15,17], m [0.1,1.0], theta0 [-1,2] deg, "
              "delta_theta [40,60] deg")
        t0 = time.perf_counter()
        pso = sw.calibrate_pso(
            theta_exp, P_exp,
            seed=args.seed,
            max_iteracoes=args.iters or 800,
            verbose=not args.quiet,
        )
        dt = time.perf_counter() - t0
        print(f"     -> erro = {pso['erro']:.6f} kPa em {pso['iteracoes']} iteracoes "
              f"({dt:.0f} s)")
        print(f"        Rc = {pso['Rc']:.6f} | m = {pso['m']:.6f} | "
              f"theta0 = {pso['theta0_deg']:.4f} deg | "
              f"delta_theta = {pso['delta_theta_deg']:.4f} deg")
        print(f"        parou por repeticao: {pso['parou_por_repeticao']}")
        # indicação de ótimo na borda do domínio
        at_bounds = [
            name for name, x, lo, hi in zip(
                ["Rc", "m", "theta0", "delta_theta"], pso["params"], sw.LOWER, sw.UPPER)
            if np.isclose(x, lo, rtol=0, atol=1e-6) or np.isclose(x, hi, rtol=0, atol=1e-6)
        ]
        if at_bounds:
            print(f"        OTIMO NO LIMITE DO DOMINIO em: {', '.join(at_bounds)}")
        print("        (referencia Mathematica: erro ~ 63.8473 kPa, "
              "Rc=15.8686, m=0.524544, theta0=2 deg, delta_theta=40 deg - "
              "na borda do dominio)")
        best, label = pso, "PSO (compatibilidade)"

    # -------------------------------------------------------- 3b. Alternativa DE
    if not args.skip_de:
        print(f"\n[3b] Metodo alternativo: differential_evolution (seed={args.seed})")
        t0 = time.perf_counter()
        de = sw.calibrate_de(theta_exp, P_exp, seed=args.seed, maxiter=300)
        dt = time.perf_counter() - t0
        print(f"     -> erro = {de['erro']:.6f} kPa em {de['iteracoes']} iteracoes "
              f"({dt:.0f} s) | sucesso: {de['success']}")
        print(f"        Rc = {de['Rc']:.6f} | m = {de['m']:.6f} | "
              f"theta0 = {de['theta0_deg']:.4f} deg | "
              f"delta_theta = {de['delta_theta_deg']:.4f} deg")
        if best is None or de["erro"] < best["erro"]:
            best, label = de, "DE (alternativo)"

    if best is None:
        # Sem calibração: gráficos com o caso de validação
        best = {"params": np.array([
            VALID_RC, VALID_M,
            np.radians(VALID_THETA0_DEG), np.radians(VALID_DELTA_THETA_DEG)]),
            "erro": erro_val, "history": []}
        label = "validacao (sem calibracao)"

    # ------------------------------------------------------------ 4. Gráficos/CSV
    Rc, m, theta0, delta_theta = best["params"]
    print(f"\n[4] Gerando graficos e exportando resultados com o melhor ajuste "
          f"({label}): erro = {best['erro']:.6f} kPa")
    res = sw.simulate_full(Rc, m, theta0, delta_theta, theta_exp, P_exp)
    figs = sw.make_plots(res, calibration=best, outdir=args.outdir, data=data)
    csv_main = sw.export_csv(res, outdir=args.outdir)
    csv_deg = sw.export_pressure_degrees_bar(
        Rc, m, theta0, delta_theta,
        theta_i=theta_exp[0], theta_f=theta_exp[-1], P1=data["P1"],
        outdir=args.outdir,
    )
    for f in figs + [csv_main, csv_deg]:
        print(f"    salvo: {f}")

    print("\nConcluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())