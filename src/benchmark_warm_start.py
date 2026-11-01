"""Runtime-only benchmark: sequential MIP warm-starting for WS and EEVS.

Solves WS and EEVS (sample-mean alpha) for a single scarcity level with
`warm_start_sequential=True` and compares total/average per-scenario Gurobi
`Runtime` against the existing baseline `solve_diagnostics.md` for the same
scarcity level (produced by a normal `run_experiments` run).

Safe by construction -- never touches baseline files:
  - Uses the same `load_base_parameters`/`scale_supply`/`sample_alpha` (same
    `seed`) as `run_experiments.run_scarcity_level` so alpha draws match the
    baseline run exactly, for a fair comparison.
  - `WS`/`EV_and_EEVS` write their warm-start CSV outputs with a `_warmstart`
    filename suffix (never the baseline filenames).
  - Writes its own `results/scarce_<scarce>_solve_diagnostics_warmstart.md`
    (a new file, does not overwrite `scarce_<scarce>_solve_diagnostics.md`).
  - Does not call `DE` or the population-mean `EV_and_EEVS` variant at all.

Usage:
    python benchmark_warm_start.py [scarce]   # default scarce = 0.75
"""

import sys

from ev_eevs_model import EV_and_EEVS
from ws_model import WS
from paths import results_path
from run_experiments import load_base_parameters, scale_supply, sample_alpha, write_solve_diagnostics_md


def run_benchmark(scarce, num_scenarios=150, seed=42):
    (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
     M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s, N_k_nod, N_k_d_nod) = load_base_parameters()

    b = scale_supply(b, K, N_k_s, scarce)

    import random
    random.seed(seed)

    OMEGA = list(range(num_scenarios))
    rho = {omega: 1 / len(OMEGA) for omega in OMEGA}
    alpha = sample_alpha(K, L_k, N_k_d, OMEGA, num_scenarios)
    alpha_mu_hat = {(i, k, l): sum(alpha[(i, k, l, omega)] * rho[omega] for omega in OMEGA) for k in K for l in L_k[k] for i in N_k_d[k]}

    common_kwargs = dict(b=b, OMEGA=OMEGA, N=N, A=A, T=T, S=S, L=L, R=R, K=K, N_star_k=N_star_k, N_k=N_k,
                          N_prime_k=N_prime_k, A_k=A_k, A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p,
                          M_plus=M_plus, M_minus=M_minus, alpha=alpha, beta=beta, gamma=gamma, g=g, f=f,
                          q=q, c=c, u=u, N_k_d=N_k_d, N_k_s=N_k_s, N_k_nod=N_k_nod, N_k_d_nod=N_k_d_nod,
                          scarce=scarce, rho=rho)

    diagnostics = []

    print(f'[warm-start benchmark] EEVS sample-mean, scarce={scarce}, warm_start_sequential=True\n')
    n0 = len(diagnostics)
    EV_and_EEVS(alpha_mu_type=alpha_mu_hat, diagnostics=diagnostics, warm_start_sequential=True, log_tag='sample_mean', **common_kwargs)
    for r in diagnostics[n0:]:
        r['label'] = r['label'] + '_sample_mean'

    print(f'\n[warm-start benchmark] WS, scarce={scarce}, warm_start_sequential=True\n')
    WS(diagnostics=diagnostics, warm_start_sequential=True, **common_kwargs)

    md_path = results_path(f'scarce_{scarce}_solve_diagnostics_warmstart.md')
    write_solve_diagnostics_md(md_path, scarce, diagnostics)
    print(f'\n[warm-start benchmark] wrote {md_path}')

    total_runtime = sum(r['runtime_s'] for r in diagnostics if r['runtime_s'] is not None)
    print(f'[warm-start benchmark] total Gurobi runtime across {len(diagnostics)} solves: {total_runtime:.3f}s')
    print('[warm-start benchmark] compare against results/scarce_'
          f'{scarce}_solve_diagnostics.md (baseline, warm_start_sequential=False) for the speedup.')


if __name__ == '__main__':
    scarce_arg = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
    run_benchmark(scarce_arg)
