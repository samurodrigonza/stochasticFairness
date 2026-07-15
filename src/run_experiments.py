"""Driver script that runs the EV/EEVS, WS and DE stochastic tdINDP models
across a set of resource-scarcity levels and reports/saves comparison metrics
(EVPI, VSS, etc.).
"""

import json
import random

import winsound

import tdINDP_parameters_build
from svi_data import SVI
from ev_eevs_model import EV_and_EEVS
from ws_model import WS
from de_model import DE
from paths import results_path


def load_base_parameters():
    """Load the static tdINDP network parameters (nodes, arcs, costs, capacities, ...)."""
    N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b = tdINDP_parameters_build.load_tdINDP_parameters()

    T = [0]
    u = tdINDP_parameters_build.find_capacities()

    N_k_d = {k: [i for i in N_k[k] if b[(i, k, k, 0)] < 0] for k in K}
    N_k_s = {k: [i for i in N_k[k] if b[(i, k, k, 0)] > 0] for k in K}
    N_k_nod = {k: [i for i in N_k[k] if b[(i, k, k, 0)] == 0] for k in K}
    N_k_d_nod = {k: list(set(N_k_d[k] + N_k_nod[k])) for k in K}

    N_prime_k, A_prime_k = tdINDP_parameters_build.disrput(0)

    # Collapse the time dimension (a single, un-disrupted time step is used).
    b = {(i, k, l): b[(i, k, l, 0)] for (i, k, l, _) in b.keys()}
    u = {(i, j, k): u[(i, j, k, 0)] for (i, j, k, _) in u.keys()}

    return (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
            M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s, N_k_nod, N_k_d_nod)


def scale_supply(b, K, N_k_s, scarce):
    """Scale every supply node's availability by `scarce` (in place)."""
    for k in K:
        for i in N_k_s[k]:
            b[(i, k, k)] = b[(i, k, k)] * scarce
    return b


def sample_alpha(K, L_k, N_k_d, OMEGA, num_scenarios):
    """Sample scenario-dependent fairness weights (alpha) for each demand node.

    Uses the SVI ranges when available, otherwise draws uniformly in [0, 1].
    Falls back to alpha = 1 for a single (deterministic) scenario.
    """
    if num_scenarios > 1:
        if SVI:
            return {(i, k, l, omega): random.uniform(SVI[k][i][0], SVI[k][i][1])
                    for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA}
        return {(i, k, l, omega): random.uniform(0, 1)
                for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA}
    return {(i, k, l, omega): 1 for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA}


def write_summary(filename, scarce, utility_type, num_scenarios, same_probs, pieces_length,
                   OF_EEVS_alpha_mu, OF_EEVS_alpha_mu_hat, OF_WS, OF_DE):
    lines = [
        f"scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{num_scenarios}_equal_probabilities_{same_probs}_nat_log_linear_pieces_length_{pieces_length}",
        f"\nOF_EEVS with population mean: {OF_EEVS_alpha_mu}",
        "",
        f"\nOF_EEVS with sample mean: {OF_EEVS_alpha_mu_hat}",
        "",
        f"OF_WS: {OF_WS}",
        "",
        f"OF_DE: {OF_DE}",
        "",
        f"Expectation of the Expected Value Solution (EEVS) with population mean: {OF_EEVS_alpha_mu}",
        "",
        f"Expectation of the Expected Value Solution (EEVS) with sample mean: {OF_EEVS_alpha_mu_hat}",
        "",
        f"Expected Value of Perfect Information (EVPI): {OF_WS - OF_DE}",
        "",
        f"Value of the Stochastic Solution (VSS) with population mean: {OF_DE - OF_EEVS_alpha_mu}",
        f"Value of the Stochastic Solution (VSS) with sample mean: {OF_DE - OF_EEVS_alpha_mu_hat}",
    ]
    with open(filename, "w") as file:
        file.write("\n".join(lines))


def write_summary_json(filename, scarce, utility_type, num_scenarios, same_probs, pieces_length,
                        OF_EV_alpha_mu, OF_EV_alpha_mu_hat, OF_EEVS_alpha_mu, OF_EEVS_alpha_mu_hat,
                        OF_WS, OF_DE):
    """Machine-readable twin of `write_summary` (plain JSON, one file per scarcity level).

    This is what plotter.ipynb should load automatically instead of hand-copying
    numbers out of the console/`.txt` summary - see the "scarce_*_summary.json"
    glob pattern used there.
    """
    summary = {
        'scarce': scarce, 'utility_type': utility_type, 'num_scenarios': num_scenarios,
        'same_probs': same_probs, 'pieces_length': pieces_length,
        'OF_EV_alpha_mu': OF_EV_alpha_mu, 'OF_EV_alpha_mu_hat': OF_EV_alpha_mu_hat,
        'OF_EEVS_alpha_mu': OF_EEVS_alpha_mu, 'OF_EEVS_alpha_mu_hat': OF_EEVS_alpha_mu_hat,
        'OF_WS': OF_WS, 'OF_DE': OF_DE,
        'EVPI': OF_WS - OF_DE,
        'VSS_alpha_mu': OF_DE - OF_EEVS_alpha_mu,
        'VSS_alpha_mu_hat': OF_DE - OF_EEVS_alpha_mu_hat,
    }
    with open(filename, "w") as file:
        json.dump(summary, file, indent=2)
    return summary


def write_solve_diagnostics_md(filename, scarce, records):
    """Write a Markdown audit table of every individual Gurobi solve's own
    diagnostics (status, runtime, gap, objective/bound, sizes) for a single
    scarcity-level run - one row per model solved (the EV solve, each EEVS
    scenario re-solve x2 alpha_mu variants, each WS scenario solve, the DE
    solve). Complements (does not replace) `scarce_*_summary.json`, which
    only carries the final headline objective values.
    """
    lines = [f"# Gurobi solve diagnostics - scarce = {scarce}", ""]
    n_flagged = sum(1 for r in records if r['flags'])
    lines.append(f"{len(records)} solves recorded, {n_flagged} flagged for review.")
    lines.append("")
    lines.append("| label | scenario | status | runtime (s) | sol_count | obj_val | obj_bound | mip_gap | #vars | #bin | #constrs | flags |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        def fmt(v, spec='{:.4g}'):
            return spec.format(v) if isinstance(v, (int, float)) else ('' if v is None else str(v))
        lines.append(
            f"| {r['label']} | {r['scenario'] if r['scenario'] is not None else ''} | {r['status_name']} | "
            f"{fmt(r['runtime_s'])} | {r['sol_count']} | {fmt(r['obj_val'])} | {fmt(r['obj_bound'])} | "
            f"{fmt(r['mip_gap'], '{:.2%}') if r['mip_gap'] is not None else ''} | {r['num_vars']} | "
            f"{r['num_bin_vars']} | {r['num_constrs']} | {r['flags']} |"
        )
    with open(filename, "w") as file:
        file.write("\n".join(lines) + "\n")
    return lines


def run_scarcity_level(scarce, utility_type='power', pieces_length=1, num_scenarios=150, same_probs=True, seed=42):
    """Run EV/EEVS (population & sample mean), WS and DE for a single scarcity level and report metrics."""
    print(f'\nscarce: {scarce}\n')

    (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
     M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s, N_k_nod, N_k_d_nod) = load_base_parameters()

    b = scale_supply(b, K, N_k_s, scarce)

    random.seed(seed)

    OMEGA = list(range(num_scenarios))
    rho = {omega: 1 / len(OMEGA) for omega in OMEGA}

    alpha = sample_alpha(K, L_k, N_k_d, OMEGA, num_scenarios)

    alpha_mu = {(i, k, l): (SVI[k][i][0] + SVI[k][i][1]) / 2 for k in K for l in L_k[k] for i in N_k_d[k]}
    alpha_mu_hat = {(i, k, l): sum(alpha[(i, k, l, omega)] * rho[omega] for omega in OMEGA) for k in K for l in L_k[k] for i in N_k_d[k]}

    common_kwargs = dict(b=b, OMEGA=OMEGA, N=N, A=A, T=T, S=S, L=L, R=R, K=K, N_star_k=N_star_k, N_k=N_k,
                          N_prime_k=N_prime_k, A_k=A_k, A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p,
                          M_plus=M_plus, M_minus=M_minus, alpha=alpha, beta=beta, gamma=gamma, g=g, f=f,
                          q=q, c=c, u=u, N_k_d=N_k_d, N_k_s=N_k_s, N_k_nod=N_k_nod, N_k_d_nod=N_k_d_nod,
                          scarce=scarce, rho=rho)

    print('Expected value population\n')
    diagnostics = []
    n0 = len(diagnostics)
    x_star_EV_alpha_mu, OF_EV_alpha_mu, x_star_EEVS_alpha_mu, OF_EEVS_alpha_mu = EV_and_EEVS(alpha_mu_type=alpha_mu, diagnostics=diagnostics, **common_kwargs)
    for r in diagnostics[n0:]:
        r['label'] = r['label'] + '_population_mean'
    print('Expected value sample\n')
    n1 = len(diagnostics)
    x_star_EV_alpha_mu_hat, OF_EV_alpha_mu_hat, x_star_EEVS_alpha_mu_hat, OF_EEVS_alpha_mu_hat = EV_and_EEVS(alpha_mu_type=alpha_mu_hat, diagnostics=diagnostics, **common_kwargs)
    for r in diagnostics[n1:]:
        r['label'] = r['label'] + '_sample_mean'
    x_star_WS, OF_WS = WS(diagnostics=diagnostics, **common_kwargs)
    x_star_DE, OF_DE = DE(x_start={}, diagnostics=diagnostics, **common_kwargs)

    print()
    print(f'OF_EEVS_population: {OF_EEVS_alpha_mu}')
    print()
    print(f'OF_EEVS_sample: {OF_EEVS_alpha_mu_hat}')
    print()
    print(f'OF_WS: {OF_WS}')
    print()
    print(f'OF_DE: {OF_DE}')
    print()
    print(f'Expectation of the Expected Value Solution (EVVS) with population mean: {OF_EEVS_alpha_mu}')
    print()
    print(f'Expectation of the Expected Value Solution (EVVS) with sample mean: {OF_EEVS_alpha_mu_hat}')
    print()
    print(f'Expected Value of Perfect Information (EVPI): {OF_WS-OF_DE}')
    print()
    print(f'Value of the Stochastic Solution (VSS) with population mean: {OF_DE-OF_EEVS_alpha_mu}')
    print(f'Value of the Stochastic Solution (VSS) with sample mean: {OF_DE-OF_EEVS_alpha_mu_hat}')
    print()

    summary_filename = results_path(f"scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{len(OMEGA)}_equal_probabilities_{same_probs}_nat_log_linear_pieces_length_{pieces_length}.txt")
    write_summary(summary_filename, scarce, utility_type, len(OMEGA), same_probs, pieces_length,
                  OF_EEVS_alpha_mu, OF_EEVS_alpha_mu_hat, OF_WS, OF_DE)

    summary_json_filename = results_path(f"scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{len(OMEGA)}_equal_probabilities_{same_probs}_nat_log_linear_pieces_length_{pieces_length}_summary.json")
    write_summary_json(summary_json_filename, scarce, utility_type, len(OMEGA), same_probs, pieces_length,
                        OF_EV_alpha_mu, OF_EV_alpha_mu_hat, OF_EEVS_alpha_mu, OF_EEVS_alpha_mu_hat, OF_WS, OF_DE)

    diagnostics_md_filename = results_path(f"scarce_{scarce}_solve_diagnostics.md")
    write_solve_diagnostics_md(diagnostics_md_filename, scarce, diagnostics)
    print(f'Saved Gurobi solve diagnostics to {diagnostics_md_filename}')

    return {
        'x_star_EV_alpha_mu': x_star_EV_alpha_mu, 'OF_EV_alpha_mu': OF_EV_alpha_mu,
        'x_star_EEVS_alpha_mu': x_star_EEVS_alpha_mu, 'OF_EEVS_alpha_mu': OF_EEVS_alpha_mu,
        'x_star_EV_alpha_mu_hat': x_star_EV_alpha_mu_hat, 'OF_EV_alpha_mu_hat': OF_EV_alpha_mu_hat,
        'x_star_EEVS_alpha_mu_hat': x_star_EEVS_alpha_mu_hat, 'OF_EEVS_alpha_mu_hat': OF_EEVS_alpha_mu_hat,
        'x_star_WS': x_star_WS, 'OF_WS': OF_WS,
        'x_star_DE': x_star_DE, 'OF_DE': OF_DE,
        'solve_diagnostics': diagnostics,
    }


def run_all(scarcity_levels=(0.75,), beep_when_done=True):
    """Run every scarcity level in `scarcity_levels` and optionally beep on completion.

    Defaults to a single scarcity level of 0.75 (75% of full network supply
    available) per the current experiment design - not a sweep.
    """
    results_by_scarcity = {}
    for scarce in scarcity_levels:
        results_by_scarcity[scarce] = run_scarcity_level(scarce)

    if beep_when_done:
        winsound.Beep(1000, 500)  # 1000 Hz for 500 ms

    return results_by_scarcity


if __name__ == '__main__':
    run_all()
