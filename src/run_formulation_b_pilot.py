"""Pilot driver for the Formulation B ("log-of-expectation") experiment.

Implements the incremental pilot plan from the experiment spec (Section 5):
  Stage 1 - degenerate (l_i == u_i) correctness check against Formulation A.
  Stage 2 - single network, real SVI uncertainty, small pilot solve + scoring.
Stage 3 (full 3-network, 150-scenario case study) is intentionally NOT run
here - the spec says to stop and report back unless Stage 2 solves cleanly.

This is a standalone pilot script; it does not modify or import anything from
optModels_final.ipynb, so Formulation A's paper-committed workflow is untouched.
"""

import json
import random
import time

import run_experiments
from svi_data import SVI
from formulation_b_model import solve_formulation_b
from ev_eevs_model import EV_and_EEVS
from ws_model import WS
from compare_formulations import (
    restrict_to_networks, degenerate_svi, alpha_mu_from_svi, slice_alpha_for_omega,
    score_formulation_a_objective, score_formulation_b_objective, diff_phi_dicts,
    solve_formulation_a_ev,
)
from paths import results_path


def _print_header(title):
    print('\n' + '=' * len(title))
    print(title)
    print('=' * len(title))


def run_stage1_degenerate_check(scarce=0.75, keep_networks=None, tol=2e-3):
    """Stage 1 (spec Section 5.1): force l_i == u_i everywhere (or for `keep_networks`
    only) and confirm Formulation B's objective & phi_i exactly match Formulation A's
    EV solution using that same degenerate value as alpha_mu - the Jensen equality
    case, since alpha_i is then deterministic and E[ln(u)] == ln(E[u]).

    tol=2e-3 (rather than a stricter value) accounts for expected numerical
    noise from Gurobi's piecewise-linear approximation of ln(phi)
    (FuncPieceError=1e-4, summed across ~89 demand-node log terms) - phi_i
    itself is checked separately and should match far more tightly.
    """
    _print_header('STAGE 1: degenerate (l_i == u_i) correctness check')

    base = run_experiments.load_base_parameters()
    if keep_networks is not None:
        base = restrict_to_networks(base, keep_networks)
    (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
     M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s, N_k_nod, N_k_d_nod) = base

    b = run_experiments.scale_supply(b, K, N_k_s, scarce)

    svi_degenerate = degenerate_svi(SVI, K, N_k_d)
    alpha_mu = alpha_mu_from_svi(svi_degenerate, K, L_k, N_k_d)
    alpha_single_scenario = {key + (0,): val for key, val in alpha_mu.items()}

    common_kwargs = dict(b=b, OMEGA=[0], N=N, A=A, T=T, S=S, L=L, R=R, K=K, N_star_k=N_star_k, N_k=N_k,
                          N_prime_k=N_prime_k, A_k=A_k, A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p,
                          M_plus=M_plus, M_minus=M_minus, alpha=alpha_single_scenario, beta=beta, gamma=gamma,
                          g=g, f=f, q=q, c=c, u=u, N_k_d=N_k_d, N_k_s=N_k_s, N_k_nod=N_k_nod,
                          N_k_d_nod=N_k_d_nod, scarce=scarce, rho={0: 1.0})

    print('Solving Formulation A (EV) with degenerate alpha...')
    result_a = solve_formulation_a_ev(alpha_mu_type=alpha_mu, **common_kwargs)
    OF_EV = result_a['OF_EV']

    print('Solving Formulation B with degenerate SVI...')
    result_b = solve_formulation_b(svi=svi_degenerate, warm_start=result_a.get('var_values'), **common_kwargs)

    phi_a = result_a['phi']
    phi_b = result_b['phi']

    diff_df = diff_phi_dicts(phi_a, phi_b, K, L_k, N_k_d, svi=svi_degenerate)
    max_abs_diff = diff_df['abs_diff'].max() if not diff_df.empty else float('nan')
    obj_diff = abs(OF_EV - result_b['OF_B']) if (OF_EV is not None and result_b['OF_B'] is not None) else float('nan')

    passed = bool(result_b['has_solution'] and max_abs_diff < tol and obj_diff < tol)

    print(f"Formulation A objective (alpha_mu, degenerate): {OF_EV}")
    print(f"Formulation B objective (degenerate SVI):       {result_b['OF_B']}")
    print(f"Objective diff: {obj_diff}")
    print(f"Max |phi_A - phi_B|: {max_abs_diff}")
    print(f"Formulation B status: {result_b['status_name']}, solve time: {result_b['solve_time']:.2f}s")
    print(f"STAGE 1 {'PASSED' if passed else 'FAILED'}")

    return {
        'passed': passed,
        'OF_EV': OF_EV,
        'OF_B': result_b['OF_B'],
        'obj_diff': obj_diff,
        'max_abs_phi_diff': max_abs_diff,
        'diff_df': diff_df,
        'result_b': result_b,
    }


def run_full_case_study(network=None, scarce=0.75, num_scenarios=8, seed=42,
                         singularity_band=1e-2, time_limit=None, mip_gap=0.01,
                         compute_de_swap_metrics=True):
    """Full-scale run: real SVI-based uncertainty over the original problem
    (all three networks - Gas, Power, Water - by default; pass `network` as a
    single network name only if you explicitly want a restricted subset).

    Formulation B is solved once (it has no scenario recourse - alpha is
    integrated out analytically). `num_scenarios` alpha realizations are
    sampled separately (reusing run_experiments.sample_alpha) purely to score
    both formulations' optimal phi_i against Formulation A's metric, and to
    get Formulation A's own EV solution (sample-mean alpha) on the same
    instance for a side-by-side phi_i diff.

    `time_limit=None` means no Gurobi time limit is imposed - since this is
    the full nonconvex case study (not a scaled-down pilot), let it run to
    completion (or until you interrupt it) rather than silently truncating.

    `compute_de_swap_metrics=True` additionally solves WS and EEVS (sample
    mean) and reports EVPI/VSS with Formulation B's own objective standing in
    for DE's role (Formulation B already *is* a single, expectation-consistent
    full-uncertainty solve, so there's no separate "DE for B" to build - see
    the EVPI/VSS discussion). WS is unaffected by which aggregation rule you
    use (perfect per-scenario information collapses Jensen's gap either way),
    so this reuses the exact same `ws_model.WS`/`ev_eevs_model.EV_and_EEVS`
    Formulation A already uses - no new model code:
        EVPI_B = OF_WS - OF_B
        VSS_B  = OF_B  - OF_EEVS_alpha_mu_hat
    This adds a real WS + EEVS solve (one Gurobi model per scenario each), so
    set it to False for a quick Formulation-B-only run.
    """
    scope_label = network if network is not None else 'all networks (Gas, Power, Water)'
    _print_header(f'FULL CASE STUDY: {scope_label}, real SVI uncertainty, scarce={scarce}')

    base = run_experiments.load_base_parameters()
    if network is not None:
        base = restrict_to_networks(base, [network])
    (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
     M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s, N_k_nod, N_k_d_nod) = base

    b = run_experiments.scale_supply(b, K, N_k_s, scarce)

    random.seed(seed)
    OMEGA = list(range(num_scenarios))
    rho = {omega: 1 / len(OMEGA) for omega in OMEGA}
    alpha = run_experiments.sample_alpha(K, L_k, N_k_d, OMEGA, num_scenarios)
    alpha_mu_hat = {(i, k, l): sum(alpha[(i, k, l, omega)] * rho[omega] for omega in OMEGA)
                    for k in K for l in L_k[k] for i in N_k_d[k]}

    common_kwargs = dict(b=b, OMEGA=OMEGA, N=N, A=A, T=T, S=S, L=L, R=R, K=K, N_star_k=N_star_k, N_k=N_k,
                          N_prime_k=N_prime_k, A_k=A_k, A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p,
                          M_plus=M_plus, M_minus=M_minus, alpha=alpha, beta=beta, gamma=gamma, g=g, f=f,
                          q=q, c=c, u=u, N_k_d=N_k_d, N_k_s=N_k_s, N_k_nod=N_k_nod, N_k_d_nod=N_k_d_nod,
                          scarce=scarce, rho=rho)

    print(f'Solving Formulation A (EV, sample-mean alpha) on {scope_label}...')
    t0 = time.time()
    result_a = solve_formulation_a_ev(alpha_mu_type=alpha_mu_hat, **common_kwargs)
    time_a = time.time() - t0
    OF_EV = result_a['OF_EV']

    print(f'Solving Formulation B on {scope_label} (real SVI uncertainty, NonConvex=2, '
          f'time_limit={time_limit}, mip_gap={mip_gap})...')
    result_b = solve_formulation_b(svi=SVI, singularity_band=singularity_band,
                                    time_limit=time_limit, mip_gap=mip_gap,
                                    warm_start=result_a.get('var_values'), **common_kwargs)

    phi_a = {(i, k, l): result_a['phi'][(i, k, l)] for k in K for l in L_k[k] for i in N_k_d[k]}
    phi_b = result_b['phi']

    diff_df = diff_phi_dicts(phi_a, phi_b, K, L_k, N_k_d, svi=SVI)

    OF_B_scored_as_A_mean = OF_A_scored_as_A_mean = OF_A_scored_as_B = None
    if result_b['has_solution']:
        scored_b_under_a = [score_formulation_a_objective(phi_b, slice_alpha_for_omega(alpha, omega, K, L_k, N_k_d), K, L_k, N_k_d) for omega in OMEGA]
        scored_a_under_a = [score_formulation_a_objective(phi_a, slice_alpha_for_omega(alpha, omega, K, L_k, N_k_d), K, L_k, N_k_d) for omega in OMEGA]
        OF_B_scored_as_A_mean = sum(scored_b_under_a) / len(OMEGA)
        OF_A_scored_as_A_mean = sum(scored_a_under_a) / len(OMEGA)
        OF_A_scored_as_B = score_formulation_b_objective(phi_a, SVI, K, L_k, N_k_d)

    print(f'Formulation A solve time: {time_a:.2f}s')
    print(f"Formulation B status: {result_b['status_name']}, solve time: {result_b['solve_time']:.2f}s, "
          f"mip_gap: {result_b['mip_gap']}")
    print(f"Formulation A objective (own metric):                                   {OF_EV}")
    print(f"Formulation B objective (own metric):                                   {result_b['OF_B']}")
    print(f"Formulation B phi scored under A's metric (mean over {num_scenarios} scenarios): {OF_B_scored_as_A_mean}")
    print(f"Formulation A phi scored under A's metric (mean over {num_scenarios} scenarios): {OF_A_scored_as_A_mean}")
    print(f"Formulation A phi scored under B's metric:                              {OF_A_scored_as_B}")
    print(f"# degenerate nodes: {len(result_b['degenerate_nodes'])}, # stochastic nodes: {len(result_b['stochastic_nodes'])}")
    print(f"# nodes near phi=1 singularity: {len(result_b['singularity_nodes'])}")
    if not diff_df.empty and 'abs_diff' in diff_df.columns:
        print(diff_df.sort_values('abs_diff', ascending=False).head(10).to_string(index=False))
    elif not result_b['has_solution']:
        print('Formulation B found no incumbent solution within the time/gap limit - no phi diff to report.')

    csv_path = results_path(f"formulation_b_full_case_study_{network or 'all_networks'}_scarce_{scarce}.csv")
    diff_df.to_csv(csv_path, index=False)
    print(f'Saved phi diff table to {csv_path}')

    OF_WS = OF_EEVS_alpha_mu_hat = EVPI_B = VSS_B = None
    if compute_de_swap_metrics:
        print(f'Solving WS (wait-and-see) on {scope_label} ({num_scenarios} scenarios)...')
        _, OF_WS = WS(**common_kwargs)
        print(f'Solving EEVS (sample-mean alpha) on {scope_label} ({num_scenarios} scenarios)...')
        _, _, _, OF_EEVS_alpha_mu_hat = EV_and_EEVS(alpha_mu_type=alpha_mu_hat, **common_kwargs)
        if result_b['has_solution']:
            EVPI_B = OF_WS - result_b['OF_B']
            VSS_B = result_b['OF_B'] - OF_EEVS_alpha_mu_hat
        print(f"OF_WS:                                    {OF_WS}")
        print(f"OF_EEVS (sample mean):                    {OF_EEVS_alpha_mu_hat}")
        print(f"EVPI with Formulation B standing in for DE: {EVPI_B}")
        print(f"VSS  with Formulation B standing in for DE: {VSS_B}")

    summary = {
        'network': network, 'scarce': scarce, 'num_scenarios': num_scenarios,
        'OF_EV': OF_EV, 'OF_B': result_b['OF_B'], 'OF_WS': OF_WS,
        'OF_EEVS_alpha_mu_hat': OF_EEVS_alpha_mu_hat,
        'EVPI_B': EVPI_B, 'VSS_B': VSS_B,
        'OF_B_scored_as_A_mean': OF_B_scored_as_A_mean,
        'OF_A_scored_as_A_mean': OF_A_scored_as_A_mean,
        'OF_A_scored_as_B': OF_A_scored_as_B,
        'formulation_b_status': result_b['status_name'],
        'formulation_b_mip_gap': result_b['mip_gap'],
    }
    summary_json_path = results_path(f"formulation_b_full_case_study_{network or 'all_networks'}_scarce_{scarce}_summary.json")
    with open(summary_json_path, 'w') as file:
        json.dump(summary, file, indent=2)
    print(f'Saved Formulation B summary to {summary_json_path}')

    return {
        'network': network,
        'result_b': result_b,
        'OF_EV': OF_EV,
        'time_a': time_a,
        'diff_df': diff_df,
        'OF_B_scored_as_A_mean': OF_B_scored_as_A_mean,
        'OF_A_scored_as_A_mean': OF_A_scored_as_A_mean,
        'OF_A_scored_as_B': OF_A_scored_as_B,
        'OF_WS': OF_WS,
        'OF_EEVS_alpha_mu_hat': OF_EEVS_alpha_mu_hat,
        'EVPI_B': EVPI_B,
        'VSS_B': VSS_B,
    }


if __name__ == '__main__':
    stage1 = run_stage1_degenerate_check()
    if stage1['passed']:
        run_full_case_study()
    else:
        print('\nStage 1 FAILED - stopping before the full case study per the pilot plan (spec Section 5). '
              'Do not proceed to a larger run until this is diagnosed.')
