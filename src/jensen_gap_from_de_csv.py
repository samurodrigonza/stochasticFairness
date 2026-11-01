"""Ex-post Formulation B evaluation directly from an already-saved DE results CSV.

No re-solve of anything - not Formulation B's nonconvex MIQCP, not even a fresh
Formulation A DE solve. This reads an existing `results/DE_scarce_*.csv`
(produced by `de_model.DE` via `run_experiments.run_all`/`run_scarcity_level`)
and computes the "Jensen gap" between:

    OF_A (expectation-of-log):   E_omega[ sum_i alpha_i,omega * ln(phi_i,omega) ]
    OF_B (log-of-expectation):   sum_i ln( E_omega[ phi_i,omega ** alpha_i,omega ] )

on the SAME allocation - Formulation A's DE solution's own per-scenario
recourse phi_i,omega. Since alpha never enters any constraint, the DE model
still lets phi_i vary by scenario (a genuine second-stage recourse decision
driven purely by how the objective weighs each scenario's alpha draw), so this
is a real Monte-Carlo sample of (phi_i,omega, alpha_i,omega) pairs per node -
enough to estimate E[phi_i**alpha_i] empirically with no closed-form-uniform
assumption and no nonconvex solve. Per Jensen's inequality (ln concave),
OF_B >= OF_A always; this script reports that gap in place of solving
Formulation B to optimality, which was found to not converge in practical
time at full 3-network scale (see docs/dossier.md, Section 11).

Usage: `python jensen_gap_from_de_csv.py` (auto-discovers the newest
`results/DE_scarce_*.csv`), or call `jensen_gap_from_de_csv(csv_path)` directly.
"""

import ast
import glob
import math

import pandas as pd

from paths import results_path
from svi_data import SVI


def _omega_columns(df):
    """The DE CSV's columns are the per-scenario values ('0', '1', ..., 'N-1')
    plus summary-stat columns (mean/std/min/max/median/mode/q1/q2/q3) appended
    by gurobi_utils.build_results_dataframe - keep only the digit-named ones,
    in scenario order.
    """
    cols = [c for c in df.columns if c.isdigit()]
    return sorted(cols, key=int)


def jensen_gap_from_de_csv(csv_path, svi=None, floor=1e-9):
    """Compute the per-node and total Jensen gap from a saved DE results CSV.

    For each demand node, phi_i,omega and alpha_i,omega are pulled straight
    from the CSV's 'phi_(...)' / 'alpha_(...)' rows (equal scenario
    probabilities are assumed, matching every DE run in this codebase so far -
    `rho = 1/num_scenarios`). Each scenario's phi_i**alpha_i and
    alpha_i*ln(phi_i) are evaluated individually, then averaged across
    scenarios - i.e. exactly "evaluate per scenario, then take the average",
    the standard Monte-Carlo estimator of an expectation.

    Returns a dict: csv_path, num_scenarios, OF_A_DE_check (should match the
    OF_DE this CSV was generated alongside, up to floating point),
    OF_B_expost, jensen_gap (= OF_B_expost - OF_A_DE_check, >= 0 by Jensen),
    and a per-node DataFrame sorted by gap size (largest first).
    """
    df = pd.read_csv(csv_path, index_col=0)
    omega_cols = _omega_columns(df)
    n_omega = len(omega_cols)
    rho = 1.0 / n_omega

    phi_rows = [idx for idx in df.index if idx.startswith('phi_(')]

    rows = []
    OF_A_check = 0.0
    OF_B_expost = 0.0
    for phi_row in phi_rows:
        key_str = phi_row[len('phi_'):]
        i, k, l = ast.literal_eval(key_str)
        alpha_row = 'alpha_' + key_str
        if alpha_row not in df.index:
            continue

        phi_vals = df.loc[phi_row, omega_cols].astype(float).clip(lower=floor).tolist()
        alpha_vals = df.loc[alpha_row, omega_cols].astype(float).tolist()

        # Evaluate per scenario, then average:
        e_log_util = rho * sum(a * math.log(p) for a, p in zip(alpha_vals, phi_vals))
        e_util = rho * sum(p ** a for a, p in zip(alpha_vals, phi_vals))
        log_e_util = math.log(e_util) if e_util > 0 else math.log(floor)

        OF_A_check += e_log_util
        OF_B_expost += log_e_util

        row = {
            'network': k, 'commodity': l, 'node': i,
            'E_log_util_A': e_log_util, 'log_E_util_B': log_e_util,
            'node_jensen_gap': log_e_util - e_log_util,
        }
        if svi is not None:
            lo, hi = svi[k][i]
            row['svi_width'] = hi - lo
        rows.append(row)

    per_node = pd.DataFrame(rows).sort_values('node_jensen_gap', ascending=False).reset_index(drop=True)

    return {
        'csv_path': csv_path,
        'num_scenarios': n_omega,
        'OF_A_DE_check': OF_A_check,
        'OF_B_expost': OF_B_expost,
        'jensen_gap': OF_B_expost - OF_A_check,
        'per_node': per_node,
    }


def find_latest_de_csv():
    """Newest `results/DE_scarce_*.csv` on disk (alphabetical == chronological
    here since the filename embeds scarce/num_scenarios, not a timestamp - if
    you have more than one DE run saved, pass the exact path explicitly instead
    of relying on this).
    """
    candidates = sorted(glob.glob(results_path('DE_scarce_*.csv')))
    if not candidates:
        raise FileNotFoundError("No results/DE_scarce_*.csv found - run the DE model (run_experiments) first.")
    return candidates[-1]


if __name__ == '__main__':
    csv_path = find_latest_de_csv()
    print(f'Reading DE solution from: {csv_path}')
    result = jensen_gap_from_de_csv(csv_path, svi=SVI)

    print(f"Scenarios: {result['num_scenarios']}")
    print(f"OF_A (DE, recomputed check):                                  {result['OF_A_DE_check']:.6f}")
    print(f"OF_B (ex-post log-of-expected-utility on DE's own allocation): {result['OF_B_expost']:.6f}")
    print(f"Jensen gap (OF_B_expost - OF_A_DE):                            {result['jensen_gap']:.6f}")
    print()
    print('Top 10 nodes by Jensen gap:')
    print(result['per_node'].head(10).to_string(index=False))

    out_csv = results_path('jensen_gap_expost_from_de.csv')
    result['per_node'].to_csv(out_csv, index=False)
    print(f"\nSaved per-node breakdown to {out_csv}")
