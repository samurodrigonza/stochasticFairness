"""Comparison/scoring utilities for the Formulation B pilot.

These help answer the "what to report back" questions from the experiment
spec (Section 6): scoring a Formulation-B-optimal allocation under
Formulation A's metric, diffing phi_i between the two formulations, and
restricting the tdINDP parameter set down to a subset of networks so pilot
runs (Section 5, Stage 2) stay cheap.
"""

import math

import pandas as pd
from gurobipy import GRB, Model, quicksum

from gurobi_utils import configure_common_gurobi_params

WIDTH_EPS = 1e-6


def restrict_to_networks(base_params, keep_networks):
    """Filter a `run_experiments.load_base_parameters()` tuple down to `keep_networks`.

    N, A, T, S, L, R, v, h, p, M_plus, M_minus, alpha, beta, g, f, q, c are passed
    through unchanged - none of the model-building code (EV/EEVS, WS, DE,
    Formulation B) actually reads them; only K, N_k, A_k, L_k, N_k_d, N_k_s,
    N_k_nod, N_k_d_nod, u, gamma and b are network-structural and get filtered.
    Cross-network `gamma` interdependency entries (referencing a dropped
    network) are dropped, which is what isolates a single network for a cheap
    pilot run.
    """
    (N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p,
     M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b, N_k_d, N_k_s, N_k_nod, N_k_d_nod) = base_params

    keep = set(keep_networks)
    missing = keep - set(K)
    if missing:
        raise ValueError(f"Unknown network(s) requested: {missing}. Available: {K}")

    K_f = [k for k in K if k in keep]
    N_k_f = {k: N_k[k] for k in K_f}
    N_star_k_f = {k: N_star_k[k] for k in K_f}
    N_prime_k_f = {k: N_prime_k[k] for k in K_f}
    A_k_f = {k: A_k[k] for k in K_f}
    A_prime_k_f = {k: A_prime_k[k] for k in K_f}
    L_k_f = {k: L_k[k] for k in K_f}
    N_k_d_f = {k: N_k_d[k] for k in K_f}
    N_k_s_f = {k: N_k_s[k] for k in K_f}
    N_k_nod_f = {k: N_k_nod[k] for k in K_f}
    N_k_d_nod_f = {k: N_k_d_nod[k] for k in K_f}
    u_f = {key: val for key, val in u.items() if key[2] in keep}
    b_f = {key: val for key, val in b.items() if key[1] in keep}
    gamma_f = {key: val for key, val in gamma.items() if key[2] in keep and key[3] in keep}

    return (N, A, T, S, L, R, K_f, N_star_k_f, N_k_f, N_prime_k_f, A_k_f, A_prime_k_f, L_k_f, v, h, p,
            M_plus, M_minus, alpha, beta, gamma_f, g, f, q, c, u_f, b_f, N_k_d_f, N_k_s_f, N_k_nod_f, N_k_d_nod_f)


def degenerate_svi(svi, K, N_k_d, at=None):
    """Build a copy of the SVI dict with every (or a chosen subset of) demand
    node's [l_i, u_i] collapsed to a single deterministic value (their midpoint
    by default), for the Stage 1 Jensen-equality correctness check.

    `at`: optional iterable of (node, network) pairs to collapse; if None,
    every demand node in every network in K is collapsed.
    """
    target = None if at is None else set(at)
    out = {k: dict(svi[k]) for k in svi}
    for k in K:
        for i in N_k_d[k]:
            if target is not None and (i, k) not in target:
                continue
            lo, hi = svi[k][i]
            mid = (lo + hi) / 2
            out[k][i] = [mid, mid]
    return out


def alpha_mu_from_svi(svi, K, L_k, N_k_d):
    """Population-mean alpha (midpoint of each node's SVI range) - the weight
    Formulation A's EV model uses; also what Formulation B's degenerate nodes
    collapse to exactly, which is the basis of the Stage 1 correctness check.
    """
    return {(i, k, l): (svi[k][i][0] + svi[k][i][1]) / 2 for k in K for l in L_k[k] for i in N_k_d[k]}


def slice_alpha_for_omega(alpha, omega, K, L_k, N_k_d):
    """Pull a single scenario's alpha realizations out of the OMEGA-indexed alpha dict."""
    return {(i, k, l): alpha[(i, k, l, omega)] for k in K for l in L_k[k] for i in N_k_d[k]}


def score_formulation_a_objective(phi, alpha_realization, K, L_k, N_k_d, floor=1e-9):
    """Recompute Formulation A's sum_i alpha_i*ln(phi_i) for a given (fixed)
    phi allocation and a single alpha realization - used to see how a
    Formulation-B-optimal allocation scores under Formulation A's metric.
    """
    total = 0.0
    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                key = (i, k, l)
                phi_val = phi.get(key)
                if phi_val is None:
                    continue
                a = alpha_realization[key]
                total += a * math.log(phi_val) if phi_val > 0 else a * math.log(floor)
    return total


def score_formulation_b_objective(phi, svi, K, L_k, N_k_d, width_eps=WIDTH_EPS, floor=1e-9):
    """Recompute Formulation B's sum_i ln(E[phi_i**alpha_i]) for a fixed phi
    allocation (e.g. Formulation A's optimal phi), for a symmetric comparison.
    """
    total = 0.0
    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                key = (i, k, l)
                phi_val = phi.get(key)
                if phi_val is None:
                    continue
                lo, hi = svi[k][i]
                width = hi - lo
                phi_val = max(phi_val, floor)
                if width < width_eps:
                    expected_util = phi_val ** lo
                elif abs(phi_val - 1.0) < 1e-9:
                    expected_util = 1.0  # analytic limit as phi -> 1
                else:
                    expected_util = (phi_val**hi - phi_val**lo) / (width * math.log(phi_val))
                expected_util = max(expected_util, floor)
                total += math.log(expected_util)
    return total


def solve_formulation_a_ev(alpha_mu_type, b, N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k,
                            v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, N_k_d, N_k_s, N_k_nod,
                            N_k_d_nod, scarce, rho, OMEGA=None, output_flag=1):
    """Self-contained Formulation A "EV" solve, for pilot comparison purposes only.

    This intentionally reproduces just the (already-working) EV block from
    `ev_eevs_model.EV_and_EEVS` - not its EEVS second half, which fixes the
    first-stage x_star and re-solves per scenario. That EEVS loop was found
    to raise `AttributeError: Unable to retrieve attribute 'x'` (the
    per-scenario second-stage model isn't reaching an optimal solution)
    independent of any Formulation B changes - it reproduces with the
    original alpha/EEVS code path too, so it's a pre-existing issue out of
    scope for this pilot. Since the pilot only needs a single-scenario
    Formulation A allocation to diff against Formulation B, this avoids that
    unrelated bug entirely rather than trying to fix EV_and_EEVS.

    Accepts the same oversized `common_kwargs`-style signature as
    EV_and_EEVS/solve_formulation_b (N, A, T, S, L, R, v, h, p, M_plus,
    M_minus, alpha, beta, g, f, q, c, OMEGA are unused) purely for
    call-compatibility.
    """
    model = Model('formulation_A_EV_reference')
    configure_common_gurobi_params(model, output_flag=output_flag)

    x_star = {(i, j, k, l): model.addVar(vtype=GRB.CONTINUOUS, name='x_star_' + str((i, j, k, l)))
              for k in K for l in L_k[k] for (i, j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
    w = {(i, k): model.addVar(vtype=GRB.BINARY, name='w_' + str((i, k))) for k in K for i in N_k[k]}
    y = {(i, j, k): model.addVar(vtype=GRB.BINARY, name='y_' + str((i, j, k))) for k in K for (i, j) in A_k[k]}
    delta_plus = {(i, k, l): model.addVar(vtype=GRB.CONTINUOUS, name='delta_plus_' + str((i, k, l)))
                  for k in K for l in L_k[k] for i in N_k_s[k]}
    model.update()

    proportional_fairness = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='proportional_fairness')
    phi = {(i, k, l): model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=abs(b[(i, k, l)]), name='phi_' + str((i, k, l)))
           for k in K for l in L_k[k] for i in N_k_d[k]}
    log_phi = {(i, k, l): model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                        ub=math.log(abs(b[(i, k, l)])) if abs(b[(i, k, l)]) > 0 else 0.0,
                                        name='log_phi_' + str((i, k, l)))
               for k in K for l in L_k[k] for i in N_k_d[k]}
    delta_minus = {(i, k, l): model.addVar(vtype=GRB.CONTINUOUS, name='delta_minus_' + str((i, k, l)))
                   for k in K for l in L_k[k] for i in N_k_d[k]}
    x = {(i, j, k, l): model.addVar(vtype=GRB.CONTINUOUS, name='x_' + str((i, j, k, l)))
         for k in K for l in L_k[k] for (i, j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}
    model.update()

    model.addConstr(proportional_fairness == quicksum(
        alpha_mu_type[(i, k, l)] * log_phi[(i, k, l)] for k in K for l in L_k[k] for i in N_k_d[k]),
        name='proportional_fairness_constraint')
    model.setObjective(proportional_fairness, GRB.MAXIMIZE)
    model.update()

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                model.addConstr(quicksum(x_star[(i, j, k, l)] for j in N_k[k] if (i, j) in A_k[k]) == b[(i, k, l)] - delta_plus[(i, k, l)],
                                 name='supply_flow_balance_' + str((i, k, l)))

    for k in K:
        for (i, j) in A_k[k]:
            if i in N_k_s[k] and j in N_k[k]:
                model.addConstr(quicksum(x_star[(i, j, k, l)] for l in L_k[k]) <= u[(i, j, k)] * y[(i, j, k)],
                                 name='arc_functionality_' + str((i, j, k)))
                model.addConstr(quicksum(x_star[(i, j, k, l)] for l in L_k[k]) <= u[(i, j, k)] * w[(i, k)],
                                 name='node_i_functionality_' + str((i, j, k)))
                model.addConstr(quicksum(x_star[(i, j, k, l)] for l in L_k[k]) <= u[(i, j, k)] * w[(j, k)],
                                 name='node_j_functionality_' + str((i, j, k)))

    for k in K:
        for k_tilde in K:
            for j in N_k[k_tilde]:
                dep_nodes = [i_dep for i_dep in N_k[k] if (i_dep,j,k,k_tilde) in gamma.keys()]
                if dep_nodes:
                    model.addConstr(quicksum(w[(i_dep,k)]*gamma[(i_dep,j,k,k_tilde)] for i_dep in dep_nodes) - w[(j,k_tilde)] >= 0,
                                    name = 'interdependent_functionality_geq_'+str((k,k_tilde,j)))

    for k in K:
        for (i, j) in A_k[k]:
            if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                model.addConstr(quicksum(x[(i, j, k, l)] for l in L_k[k]) <= u[(i, j, k)] * y[(i, j, k)],
                                 name='arc_functionality_' + str((i, j, k)))
                model.addConstr(quicksum(x[(i, j, k, l)] for l in L_k[k]) <= u[(i, j, k)] * w[(i, k)],
                                 name='node_i_functionality_' + str((i, j, k)))
                model.addConstr(quicksum(x[(i, j, k, l)] for l in L_k[k]) <= u[(i, j, k)] * w[(j, k)],
                                 name='node_j_functionality_' + str((i, j, k)))

    # Direction-exclusivity: every directed arc gets its own z (first-stage z_star for
    # x_star supply arcs, second-stage z for x demand/transshipment arcs); z_ij + z_ji <= 1
    # stops both directions of the same pair from carrying flow at once. This mirrors
    # ev_eevs_model.py's EV block so warm-starting Formulation B from this solution can
    # match variable names 1:1.
    z_star = {(i, j, k, l): model.addVar(vtype=GRB.BINARY, name='z_star_' + str((i, j, k, l)))
              for k in K for l in L_k[k] for (i, j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
    z = {(i, j, k, l): model.addVar(vtype=GRB.BINARY, name='z_' + str((i, j, k, l)))
         for k in K for l in L_k[k] for (i, j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}
    model.update()
    for k in K:
        for (i, j) in A_k[k]:
            if i in N_k_s[k] and j in N_k[k]:
                for l in L_k[k]:
                    model.addConstr(x_star[(i, j, k, l)] <= u[(i, j, k)] * z_star[(i, j, k, l)], name='dir_excl_x_star_' + str((i, j, k, l)))
                    if i != j and (j, i, k, l) in z_star:
                        model.addConstr(z_star[(i, j, k, l)] + z_star[(j, i, k, l)] <= 1, name='dir_excl_x_star_pair_' + str((i, j, k, l)))
    for k in K:
        for l in L_k[k]:
            for (i, j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    model.addConstr(x[(i, j, k, l)] <= u[(i, j, k)] * z[(i, j, k, l)], name='dir_excl_x_' + str((i, j, k, l)))
                    if i != j and (j, i, k, l) in z:
                        model.addConstr(z[(i, j, k, l)] + z[(j, i, k, l)] <= 1, name='dir_excl_x_pair_' + str((i, j, k, l)))

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                model.addConstr(quicksum(x[(i, j, k, l)] for j in N_k_d_nod[k] if (i, j) in A_k[k]) -
                                 (quicksum(x[(j, i, k, l)] for j in N_k_d_nod[k] if (j, i) in A_k[k]) +
                                  quicksum(x_star[(j, i, k, l)] for j in N_k_s[k] if (j, i) in A_k[k])) == b[(i, k, l)] + delta_minus[(i, k, l)],
                                 name='demand_flow_balance_' + str((i, k, l)))
                model.addConstr(phi[(i, k, l)] == abs(b[(i, k, l)]) - delta_minus[(i, k, l)], name='capture_phi_' + str((i, k, l)))
                model.addGenConstrLog(phi[(i, k, l)], log_phi[(i, k, l)], name='log_phi_constr_' + str((i, k, l)))

            for i in N_k_nod[k]:
                model.addConstr(quicksum(x[(i, j, k, l)] for j in N_k_d_nod[k] if (i, j) in A_k[k]) -
                                 (quicksum(x[(j, i, k, l)] for j in N_k_d_nod[k] if (j, i) in A_k[k]) +
                                  quicksum(x_star[(j, i, k, l)] for j in N_k_s[k] if (j, i) in A_k[k])) == 0,
                                 name='transshipment_flow_balance_' + str((i, k, l)))

    model.update()
    model.optimize()

    has_solution = model.SolCount > 0
    phi_values = {key: phi[key].X for key in phi} if has_solution else {}
    var_values = {var.varName: var.X for var in model.getVars()} if has_solution else {}
    OF_EV = None
    if has_solution:
        OF_EV = sum(alpha_mu_type[(i, k, l)] * math.log(phi_values[(i, k, l)]) if phi_values[(i, k, l)] > 0
                    else alpha_mu_type[(i, k, l)] * math.log(1e-9)
                    for k in K for l in L_k[k] for i in N_k_d[k])

    return {
        'status': model.status,
        'has_solution': has_solution,
        'solve_time': model.Runtime,
        'OF_EV': OF_EV,
        'phi': phi_values,
        'var_values': var_values,
    }


def diff_phi_dicts(phi_a, phi_b, K, L_k, N_k_d, svi=None, label_a='phi_A', label_b='phi_B'):
    """Per-node phi_i comparison between two solutions (e.g. Formulation A vs B)."""
    rows = []
    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                key = (i, k, l)
                pa = phi_a.get(key)
                pb = phi_b.get(key)
                row = {'network': k, 'commodity': l, 'node': i, label_a: pa, label_b: pb}
                if pa is not None and pb is not None:
                    row['diff_B_minus_A'] = pb - pa
                    row['abs_diff'] = abs(pb - pa)
                if svi is not None:
                    lo, hi = svi[k][i]
                    row['svi_width'] = hi - lo
                rows.append(row)
    df = pd.DataFrame(rows)
    if 'svi_width' in df.columns and 'abs_diff' in df.columns:
        df = df.sort_values('svi_width')
    return df
