"""Compare DE solver configurations on one identical scenario set.

Isolates the deterministic-equivalent solve, which is the only stage where the
piecewise-linear and outer-approximation treatments of ln(phi) disagree on cost:

  pwl        FuncNonlinear=0 -- closes to a 0.0000% gap, but its allocation is
             optimal for the *approximated* log (errors ~5e-04 per term).
  outer      FuncNonlinear=1 -- exact log, but its spatial bound stalls, so it
             can only be certified to `mip_gap`.
  outer_ws   as `outer`, seeded with `pwl`'s allocation as a MIP start. If the
             objective does not improve on `pwl`, that is evidence the
             allocation is optimal despite the weak bound.

OF_DE is recomputed in exact Python math.log from the solved phi in every case
(see de_model), so the reported objectives are directly comparable.
"""

import json
import time

from de_model import DE
from paths import results_path
from run_experiments import load_base_parameters, sample_alpha, scale_supply

SCARCE = 0.75
NUM_SCENARIOS = 150
SEED = 42


def main():
    (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
     M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s,
     N_k_nod, N_k_d_nod) = load_base_parameters()

    b = scale_supply(b, K, N_k_s, SCARCE)
    OMEGA = list(range(NUM_SCENARIOS))
    alpha = sample_alpha(K, L_k, N_k_d, OMEGA, NUM_SCENARIOS)
    rho = {omega: 1.0 / NUM_SCENARIOS for omega in OMEGA}

    common = dict(b=b, OMEGA=OMEGA, N=N, A=A, T=T, S=S, L=L, R=R, K=K,
                  N_star_k=N_star_k, N_k=N_k, N_prime_k=N_prime_k, A_k=A_k,
                  A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p, M_plus=M_plus,
                  M_minus=M_minus, alpha=alpha, beta=beta, gamma=gamma, g=g,
                  f=f, q=q, c=c, u=u, N_k_d=N_k_d, N_k_s=N_k_s,
                  N_k_nod=N_k_nod, N_k_d_nod=N_k_d_nod, scarce=SCARCE, rho=rho)

    results = {}

    print('\n=== [1/4] DE, piecewise-linear (FuncNonlinear=0, exact gap) ===')
    t0 = time.time()
    x_pwl, of_pwl = DE(func_nonlinear=0, mip_gap=None, log_suffix='_cfg_pwl', **common)
    results['pwl'] = (of_pwl, time.time() - t0)

    print('\n=== [2/4] DE, outer approximation (FuncNonlinear=1, MIPGap=1e-3) ===')
    t0 = time.time()
    _, of_outer = DE(func_nonlinear=1, mip_gap=1e-3, log_suffix='_cfg_outer', **common)
    results['outer_gap1e-3'] = (of_outer, time.time() - t0)

    print('\n=== [3/4] DE, outer approximation warm-started from the PWL solution ===')
    t0 = time.time()
    _, of_ws = DE(func_nonlinear=1, mip_gap=1e-3, mip_start=x_pwl,
                  log_suffix='_cfg_outer_ws', **common)
    results['outer_ws_gap1e-3'] = (of_ws, time.time() - t0)

    # Does the relaxed numerics setting let the spatial bound close at the tight
    # gap that previously stalled? Warm-started so the incumbent side is free and
    # any remaining time is spent purely on moving the bound.
    print('\n=== [4/4] DE, outer approximation + warm start at the tight MIPGap=1e-4 ===')
    t0 = time.time()
    _, of_ws_tight = DE(func_nonlinear=1, mip_gap=1e-4, mip_start=x_pwl,
                        log_suffix='_cfg_outer_ws_tight', **common)
    results['outer_ws_gap1e-4'] = (of_ws_tight, time.time() - t0)

    print('\n' + '=' * 72)
    print(f"{'config':12s} {'OF_DE (exact rescore)':>24s} {'runtime':>12s}")
    for name, (of, secs) in results.items():
        print(f'{name:12s} {of:24.12f} {secs:11.1f}s')

    base = results['pwl'][0]
    print('\ndeltas vs pwl:')
    for name, (of, _) in results.items():
        print(f'  {name:12s} {of - base:+.3e}')

    out = results_path('de_solver_config_comparison.json')
    with open(out, 'w') as fh:
        json.dump({k: {'OF_DE': v[0], 'runtime_s': v[1]} for k, v in results.items()}, fh, indent=2)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
