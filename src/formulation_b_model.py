"""Formulation B (pilot): log-of-expected-utility objective for power-utility
proportional fairness, per the "Log-of-Expectation Objective" experiment spec.

Formulation A (the paper's primary result, in ev_eevs_model.py / ws_model.py /
de_model.py) maximizes E[alpha_i] * ln(phi_i) or its per-scenario analogues.
This module instead maximizes

    sum_i  ln( E_alpha_i[ phi_i ** alpha_i ] )

where alpha_i ~ Uniform(l_i, u_i) with (l_i, u_i) = SVI[k][i]. Since alpha_i is
integrated out analytically (closed form for a uniform exponent), this is a
*single*, non-scenario model - there is no OMEGA/recourse loop here, unlike DE.

alpha / rho / OMEGA are accepted (for call-signature compatibility with the
`common_kwargs` dict built in run_experiments.py) but intentionally unused.

Known hazards handled here (see experiment spec Sections 2 and 4):
- Nodes with (u_i - l_i) < WIDTH_EPS are treated as deterministic (alpha_i is
  effectively a constant), and use the linear shortcut
  ln(E[phi_i**alpha_i]) == l_i * ln(phi_i) directly - this both avoids a 0/0
  division and, when *every* node is degenerate, makes Formulation B reduce
  exactly to Formulation A (used as the Stage 1 correctness check).
- Nodes with real uncertainty use the closed-form
  E[phi_i**alpha_i] = (phi_i**u_i - phi_i**l_i) / ((u_i - l_i) * ln(phi_i)),
  built via two `addGenConstrPow` calls, a bilinear equality constraint (this
  Gurobi version has no addGenConstrNL, so NonConvex=2 handles the bilinear
  term per the spec's documented fallback), and a final `addGenConstrLog`.
- The phi_i -> 1 singularity is not eliminated (that would need indicator/SOS
  branching not implemented here), but expected_util_i is floored at 1e-6 and
  every node whose optimal phi_i lands within `singularity_band` of 1 is
  logged and reported, per the spec's "log every instance" requirement.
"""

import math

from gurobipy import GRB, Model, quicksum

from gurobi_utils import configure_common_gurobi_params
from svi_data import SVI

WIDTH_EPS = 1e-6           # SVI (u_i - l_i) below this is treated as deterministic
EXPECTED_UTIL_LB = 1e-6    # floor recommended in the spec (Section 4)

_STATUS_NAMES = {
    getattr(GRB, name): name
    for name in ('OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'TIME_LIMIT', 'SUBOPTIMAL', 'NUMERIC')
    if hasattr(GRB, name)
}


def _status_name(status_code):
    return _STATUS_NAMES.get(status_code, f'STATUS_{status_code}')


def solve_formulation_b(b, N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, N_k_d, N_k_s, N_k_nod, N_k_d_nod, scarce, rho, OMEGA=None,
                         svi=None, singularity_band=1e-2, width_eps=WIDTH_EPS, output_flag=1, time_limit=None, mip_gap=0.01, warm_start=None):
    """Build and solve the single-scenario Formulation B model.

    `warm_start`: optional dict of {varName: value} (e.g. `solve_formulation_a_ev(...)['var_values']`)
    used as a Gurobi MIP start. Network/flow variables shared with Formulation A's EV
    model (x_star, w, y, delta_plus, x, z_star, z) are set directly by name match;
    phi/delta_minus and the expected-utility-chain variables that only exist here
    (log_phi, p_hi, p_lo, expected_util, log_expected_util) are all re-derived
    analytically from the same floored phi_i per node (never mixed with the raw,
    possibly-zero warm_start value), so every general/bilinear constraint in the chain
    sees a mutually consistent Start point instead of getting rejected as infeasible.
    `mip_gap` defaults to 1% (loose on purpose - see experiment spec, this isn't worth
    stressing over a 1e-4 gap).

    Returns a dict with the solved model's status/timing, the raw variable
    values, the per-node phi_i solution, and diagnostics (which nodes were
    treated as degenerate vs. stochastic, and which landed near the phi_i=1
    singularity).
    """
    if svi is None:
        svi = SVI

    print('Formulation B: log-of-expected-utility (pilot)')
    model = Model('formulation_B_tdINDP')
    configure_common_gurobi_params(model, output_flag=output_flag)
    model.setParam('NonConvex', 2)
    # These models are struggling to find *any* incumbent within the default
    # MIP strategy (observed: 5+ hours / 400k+ nodes with zero incumbent on
    # the full 3-network case study). MIPFocus=1 tells Gurobi to prioritize
    # finding feasible solutions over proving optimality/improving the bound;
    # Heuristics/NoRelHeurTime spend more effort on rounding/diving heuristics
    # up front, which matters a lot when the LP relaxation bound is this weak
    # (89 quadratic constraints + 316 SOS constraints after presolve).
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.5)
    model.setParam('NoRelHeurTime', 30)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    if mip_gap is not None:
        model.setParam('MIPGap', mip_gap)

    '''
    first stage variables (identical to Formulation A)
    '''

    x_star = {(i,j,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_star_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
    w = {(i,k): model.addVar(vtype = GRB.BINARY, name = 'w_'+str((i,k))) for k in K for i in N_k[k]}
    y = {(i,j,k): model.addVar(vtype = GRB.BINARY, name = 'y_'+str((i,j,k))) for k in K for (i,j) in A_k[k]}
    delta_plus = {(i,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_plus_'+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_s[k]}
    model.update()

    '''
    second stage variables (phi/delta_minus/x identical to Formulation A;
    the expected-utility chain below replaces Formulation A's alpha*log_phi objective term)
    '''

    phi = {(i,k,l):model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub = abs(b[(i,k,l)]),name="phi_"+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_d[k]}
    delta_minus = {(i,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_minus_'+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_d[k]}
    x = {(i,j,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}

    model.update()

    '''
    Expected-utility chain, per demand node (this is the part that differs from Formulation A)
    '''

    log_phi = {}
    expected_util = {}
    log_expected_util = {}
    degenerate_nodes = []
    stochastic_nodes = []
    # Names whose Start value is set here from a *consistent* (jointly-derived)
    # phi_warm, so the later generic name-matching pass must NOT clobber them
    # with the raw (unfloored) warm_start value - see note below.
    consistent_warm_names = set()

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                lo, hi = svi[k][i]
                width = hi - lo
                b_i = abs(b[(i,k,l)])
                log_phi_ub = math.log(b_i) if b_i > 0 else 0.0

                log_phi[(i,k,l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=log_phi_ub, name="log_phi_"+str((i,k,l)))
                model.addGenConstrLog(phi[(i,k,l)], log_phi[(i,k,l)], name = 'log_phi_constr_'+str((i,k,l)))

                # Warm start: derive this node's expected-utility-chain Start values from
                # the EEV/EV solution's phi_i (Formulation A has no p_hi/p_lo/expected_util/
                # log_expected_util of its own - these have to be computed here analytically).
                #
                # IMPORTANT: phi_warm gets floored away from 0 (ln(0) is undefined) before
                # deriving log_phi_warm/p_hi_warm/etc. If we then let the raw, unfloored
                # warm_start['phi_...'] value (which can be exactly 0) get assigned to
                # phi.Start separately, phi.Start and log_phi.Start become mutually
                # inconsistent with the addGenConstrLog general constraint by a wide margin.
                # That kind of internal inconsistency, repeated across many demand nodes, is
                # the most likely reason Gurobi's MIP-start repair heuristic was rejecting the
                # whole start ("did not produce a new incumbent") on the real run. Fix: always
                # set phi.Start (and delta_minus.Start, via the flow-balance identity
                # phi = |b_i| - delta_minus) from the SAME floored phi_warm used everywhere
                # else in this node's chain, and protect it from being overwritten below.
                phi_warm = warm_start.get('phi_'+str((i,k,l))) if warm_start else None
                if phi_warm is not None:
                    phi_warm = max(phi_warm, 1e-9)
                    log_phi_warm = math.log(phi_warm)
                    log_phi[(i,k,l)].Start = log_phi_warm
                    phi[(i,k,l)].Start = phi_warm
                    delta_minus[(i,k,l)].Start = b_i - phi_warm
                    consistent_warm_names.add('log_phi_'+str((i,k,l)))
                    consistent_warm_names.add('phi_'+str((i,k,l)))
                    consistent_warm_names.add('delta_minus_'+str((i,k,l)))

                if width < width_eps:
                    # Deterministic exponent (l_i == u_i): ln(E[phi**alpha]) == l_i*ln(phi) exactly.
                    # No pow/bilinear machinery needed, and no 0/0 hazard.
                    degenerate_nodes.append((i,k,l))
                    log_expected_util[(i,k,l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='log_expected_util_'+str((i,k,l)))
                    model.addConstr(log_expected_util[(i,k,l)] == lo*log_phi[(i,k,l)], name='degenerate_log_expected_util_'+str((i,k,l)))
                    if phi_warm is not None:
                        log_expected_util[(i,k,l)].Start = lo*log_phi_warm
                        consistent_warm_names.add('log_expected_util_'+str((i,k,l)))
                else:
                    stochastic_nodes.append((i,k,l))
                    # For phi in [0, b_i] and alpha in [l_i,u_i] subset [0,1] (SVI convention),
                    # phi**alpha is bounded above by max(1, b_i); this is a safe, finite ub.
                    ub_eu = max(1.0, b_i)

                    p_hi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub_eu, name='p_hi_'+str((i,k,l)))
                    p_lo = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub_eu, name='p_lo_'+str((i,k,l)))

                    if hi == 0:
                        model.addConstr(p_hi == 1, name='p_hi_fixed_'+str((i,k,l)))
                    else:
                        model.addGenConstrPow(phi[(i,k,l)], p_hi, hi, name='p_hi_pow_'+str((i,k,l)))

                    if lo == 0:
                        model.addConstr(p_lo == 1, name='p_lo_fixed_'+str((i,k,l)))
                    else:
                        model.addGenConstrPow(phi[(i,k,l)], p_lo, lo, name='p_lo_pow_'+str((i,k,l)))

                    expected_util[(i,k,l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=EXPECTED_UTIL_LB, ub=ub_eu, name='expected_util_'+str((i,k,l)))
                    # expected_util_i * (u_i-l_i) * log_phi_i == p_hi - p_lo   (bilinear -> NonConvex=2)
                    model.addConstr(expected_util[(i,k,l)] * log_phi[(i,k,l)] * width == p_hi - p_lo, name='expected_util_def_'+str((i,k,l)))

                    log_expected_util[(i,k,l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=math.log(ub_eu), name='log_expected_util_'+str((i,k,l)))
                    model.addGenConstrLog(expected_util[(i,k,l)], log_expected_util[(i,k,l)], name='log_expected_util_constr_'+str((i,k,l)))

                    if phi_warm is not None:
                        p_hi_warm = phi_warm**hi
                        p_lo_warm = phi_warm**lo
                        if abs(phi_warm - 1.0) < 1e-9:
                            expected_util_warm = 1.0  # analytic limit as phi -> 1
                        else:
                            expected_util_warm = (p_hi_warm - p_lo_warm) / (width*log_phi_warm)
                        expected_util_warm = max(expected_util_warm, EXPECTED_UTIL_LB)
                        p_hi.Start = p_hi_warm
                        p_lo.Start = p_lo_warm
                        expected_util[(i,k,l)].Start = expected_util_warm
                        log_expected_util[(i,k,l)].Start = math.log(expected_util_warm)
                        consistent_warm_names.add('p_hi_'+str((i,k,l)))
                        consistent_warm_names.add('p_lo_'+str((i,k,l)))
                        consistent_warm_names.add('expected_util_'+str((i,k,l)))
                        consistent_warm_names.add('log_expected_util_'+str((i,k,l)))

    model.update()

    model.setObjective(quicksum(log_expected_util[key] for key in log_expected_util), GRB.MAXIMIZE)
    model.update()

    '''

    Constraints (identical to Formulation A's single-scenario / EV block - untouched network/flow portion)

    '''

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                model.addConstr(quicksum(x_star[(i,j,k,l)] for j in N_k[k] if (i,j) in A_k[k]) == b[(i,k,l)] - delta_plus[(i,k,l)], name = 'supply_flow_balance_'+str((i,k,l)))

    for k in K:
        for (i,j) in A_k[k]:
            if i in N_k_s[k] and j in N_k[k]:
                model.addConstr(quicksum(x_star[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*y[(i,j,k)], name = 'arc_functionality_'+str((i,j,k)))
                model.addConstr(quicksum(x_star[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*w[(i,k)], name = 'node_i_functionality_'+str((i,j,k)))
                model.addConstr(quicksum(x_star[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*w[(j,k)], name = 'node_j_functionality_'+str((i,j,k)))

    for k in K:
        for k_tilde in K:
            for j in N_k[k_tilde]:
                dep_nodes = [i_dep for i_dep in N_k[k] if (i_dep,j,k,k_tilde) in gamma.keys()]
                if dep_nodes:
                    model.addConstr(quicksum(w[(i_dep,k)]*gamma[(i_dep,j,k,k_tilde)] for i_dep in dep_nodes) - w[(j,k_tilde)] >= 0,
                                    name = 'interdependent_functionality_geq_'+str((k,k_tilde,j)))

    for k in K:
        for (i,j) in A_k[k]:
            if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                model.addConstr(quicksum(x[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*y[(i,j,k)], name = 'arc_functionality_'+str((i,j,k)))
                model.addConstr(quicksum(x[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*w[(i,k)], name = 'node_i_functionality_'+str((i,j,k)))
                model.addConstr(quicksum(x[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*w[(j,k)], name = 'node_j_functionality_'+str((i,j,k)))

    # Direction-exclusivity: every directed arc gets its own z (first-stage z_star for
    # x_star supply arcs, second-stage z for x demand/transshipment arcs); z_ij + z_ji <= 1
    # stops both directions of the same pair from carrying flow at once.
    z_star = {(i,j,k,l): model.addVar(vtype = GRB.BINARY, name = 'z_star_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
    z = {(i,j,k,l): model.addVar(vtype = GRB.BINARY, name = 'z_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}
    model.update()
    for k in K:
        for (i,j) in A_k[k]:
            if i in N_k_s[k] and j in N_k[k]:
                for l in L_k[k]:
                    model.addConstr(x_star[(i,j,k,l)] <= u[(i,j,k)]*z_star[(i,j,k,l)], name = 'dir_excl_x_star_'+str((i,j,k,l)))
                    if i != j and (j,i,k,l) in z_star:
                        model.addConstr(z_star[(i,j,k,l)] + z_star[(j,i,k,l)] <= 1, name = 'dir_excl_x_star_pair_'+str((i,j,k,l)))
    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    model.addConstr(x[(i,j,k,l)] <= u[(i,j,k)]*z[(i,j,k,l)], name = 'dir_excl_x_'+str((i,j,k,l)))
                    if i != j and (j,i,k,l) in z:
                        model.addConstr(z[(i,j,k,l)] + z[(j,i,k,l)] <= 1, name = 'dir_excl_x_pair_'+str((i,j,k,l)))

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                model.addConstr(quicksum(x[(i,j,k,l)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == b[(i,k,l)] + delta_minus[(i,k,l)], name = 'demand_flow_balance_'+str((i,k,l)))
                model.addConstr(phi[(i,k,l)] == abs(b[(i,k,l)]) - delta_minus[(i,k,l)], name = 'capture_phi_'+str((i,k,l)))

            for i in N_k_nod[k]:
                model.addConstr(quicksum(x[(i,j,k,l)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == 0, name = 'transshipment_flow_balance_'+str((i,k,l)))

    model.update()

    if warm_start:
        # Complete the MIP start: flow/functionality/direction-exclusivity variables
        # (x_star, w, y, delta_plus, x, z_star, z) share exact variable names with
        # Formulation A's EV model since the network/flow portion is untouched, so a
        # straight name match covers them all. phi/delta_minus/log_phi/p_hi/p_lo/
        # expected_util/log_expected_util are skipped here - they were already set
        # above from a mutually-consistent, jointly-derived phi_warm per node
        # (see consistent_warm_names note above); re-assigning them from the raw
        # warm_start dict would reintroduce the ln(0)-vs-0 style inconsistency that
        # likely caused Gurobi to reject the MIP start outright on the full run.
        for var in model.getVars():
            if var.varName in warm_start and var.varName not in consistent_warm_names:
                var.Start = warm_start[var.varName]

    model.optimize()

    status = model.status
    status_name = _status_name(status)
    has_solution = model.SolCount > 0

    phi_values = {key: phi[key].X for key in phi} if has_solution else {}
    x_star_values = {var.varName: var.x for var in model.getVars()} if has_solution else {}
    OF_B = model.ObjVal if has_solution else None

    singularity_nodes = []
    if has_solution:
        for key, val in phi_values.items():
            if abs(val - 1.0) < singularity_band:
                singularity_nodes.append((key, val))
        if singularity_nodes:
            print(f'[Formulation B] {len(singularity_nodes)} node(s) landed within {singularity_band} of the phi=1 singularity:')
            for key, val in singularity_nodes:
                print(f'    phi{key} = {val}')

    return {
        'status': status,
        'status_name': status_name,
        'has_solution': has_solution,
        'solve_time': model.Runtime,
        'mip_gap': getattr(model, 'MIPGap', None) if has_solution else None,
        'OF_B': OF_B,
        'phi': phi_values,
        'x_star_B': x_star_values,
        'degenerate_nodes': degenerate_nodes,
        'stochastic_nodes': stochastic_nodes,
        'singularity_nodes': singularity_nodes,
    }
