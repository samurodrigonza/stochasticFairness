"""Wait-and-See (WS) model: solve each scenario independently with perfect information."""

import math

from gurobipy import GRB, Model, quicksum

from gurobi_utils import configure_common_gurobi_params, save_results_csv, collect_solve_diagnostics
from paths import results_path


def WS(b, OMEGA, N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, N_k_d, N_k_s, N_k_nod, N_k_d_nod, scarce, rho, x_start = {}, diagnostics=None, warm_start_sequential=False):
    """`diagnostics`: optional list; if provided, one solve-diagnostics record
    per scenario is appended to it - see `run_experiments.write_solve_diagnostics_md`.

    `warm_start_sequential`: runtime-only experiment, off by default (does not
    change results). When True, each scenario solve after the first is given a
    Gurobi MIP start built from the previous scenario's full solution (all
    vars: x_star, w, y, z_star, delta_plus, phi, log_phi, delta_minus, x, z).
    This is valid because `alpha` (the only thing that varies by scenario) only
    ever appears in the objective, never in a constraint, so every scenario's
    WS model shares the exact same feasible region -- the previous scenario's
    optimum is provably feasible for the next one. Only intended for a
    side-by-side runtime comparison against a baseline
    (`warm_start_sequential=False`) run; output filenames get a `_warmstart`
    suffix so they never collide with/overwrite baseline results.
    """

    x_star_WS = {}
    OF_omega = []
    prev_solution = None  # carries the last scenario's full solution for warm_start_sequential
    for omega in OMEGA:
        print(f'Wait and see, scenario {omega+1} out of {len(OMEGA)}')

        model = Model('stochastic_tdINDP')

        '''
        Gurobi parameters

        '''
        utility_type = 'power'
        pieces_length = 1
        configure_common_gurobi_params(model)

        '''
        first stage variables
        '''

        x_star = {(i,j,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_star_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
        w = {(i,k): model.addVar(vtype = GRB.BINARY, name = 'w_'+str((i,k))) for k in K for i in N_k[k]}
        y = {(i,j,k): model.addVar(vtype = GRB.BINARY, name = 'y_'+str((i,j,k))) for k in K for (i,j) in A_k[k]}
        delta_plus = {(i,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_plus_'+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_s[k]}
        model.update()

        '''
        second stage variables
        '''

        proportional_fairness = model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'proportional_fairness_'+str(omega))
        phi = {(i,k,l,omega):model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub = abs(b[(i,k,l)]),name="phi_"+str((i,k,l,omega))) for k in K for l in L_k[k] for i in N_k_d[k]}
        log_phi = {(i,k,l,omega):model.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,ub=math.log(abs(b[(i,k,l)])), name="log_phi_"+str((i,k,l,omega))) for k in K for l in L_k[k] for i in N_k_d[k]}
        delta_minus = {(i,k,l,omega): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_minus_'+str((i,k,l,omega))) for k in K for l in L_k[k] for i in N_k_d[k]}
        x = {(i,j,k,l,omega): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_'+str((i,j,k,l,omega))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}

        model.update()

        '''
        Objective functions:

        '''
        if utility_type == 'linear':
            # Linear
            model.addConstr(proportional_fairness == quicksum(log_phi[(i,k,l,omega)] + math.log(alpha[(i,k,l,omega)]) for k in K for l in L_k[k] for i in N_k_d[k]), name = 'proportional_fairness_constraint')
        else:
            # Power
            model.addConstr(proportional_fairness == quicksum(alpha[(i,k,l,omega)]*log_phi[(i,k,l,omega)] for k in K for l in L_k[k] for i in N_k_d[k]), name = 'proportional_fairness_constraint')

        model.setObjective(proportional_fairness, GRB.MAXIMIZE)
        model.update()

        '''

        Constraints

        '''

        for k in K:
            for l in L_k[k]:
                for i in N_k_s[k]:
                    model.addConstr(quicksum(x_star[(i,j,k,l)] for j in N_k[k] if (i,j) in A_k[k]) - quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k]) == b[(i,k,l)] - delta_plus[(i,k,l)], name = 'supply_flow_balance_'+str((i,k,l)))


        for k in K:
            for (i,j) in A_k[k]:
                if i in N_k_s[k] and j in N_k[k]:
                    model.addConstr(quicksum(x_star[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*y[(i,j,k)], name = 'arc_functionality_'+str((i,j,k)))
                    model.addConstr(quicksum(x_star[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*w[(i,k)], name = 'node_i_functionality_'+str((i,j,k)))
                    model.addConstr(quicksum(x_star[(i,j,k,l)] for l in L_k[k]) <= u[(i,j,k)]*w[(j,k)], name = 'node_j_functionality_'+str((i,j,k)))


        for k in K:
            for k_tilde in K:
                for j in N_k[k_tilde]:
                    if (i,j,k,k_tilde) in gamma.keys():
                        model.addConstr(quicksum(w[(i,k)]*gamma[(i,j,k,k_tilde)] for i in N_k[k]) <= w[(j,k_tilde)], name = 'interdependent_functionality_'+str((k,k_tilde,j)))



        for k in K:
            for (i,j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*y[(i,j,k)], name = 'arc_functionality_'+str((i,j,k,omega)))
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*w[(i,k)], name = 'node_i_functionality_'+str((i,j,k,omega)))
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*w[(j,k)], name = 'node_j_functionality_'+str((i,j,k,omega)))

        # Direction-exclusivity: every directed arc gets its own z (first-stage z_star for x_star
        # supply arcs, second-stage z for x demand/transshipment arcs); z_ij + z_ji <= 1 stops both
        # directions of the same pair from carrying flow at once.
        z_star = {(i,j,k,l): model.addVar(vtype = GRB.BINARY, name = 'z_star_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
        z = {(i,j,k,l): model.addVar(vtype = GRB.BINARY, name = 'z_'+str((i,j,k,l,omega))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}
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
                        model.addConstr(x[(i,j,k,l,omega)] <= u[(i,j,k)]*z[(i,j,k,l)], name = 'dir_excl_x_'+str((i,j,k,l,omega)))
                        if i != j and (j,i,k,l) in z:
                            model.addConstr(z[(i,j,k,l)] + z[(j,i,k,l)] <= 1, name = 'dir_excl_x_pair_'+str((i,j,k,l,omega)))

        for k in K:
            for l in L_k[k]:
                for i in N_k_d[k]:
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l,omega)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == b[(i,k,l)] + delta_minus[(i,k,l,omega)], name = 'demand_flow_balance_'+str((i,k,l,omega)))
                    model.addConstr(phi[(i,k,l,omega)] == abs(b[(i,k,l)]) - delta_minus[(i,k,l,omega)], name = 'capture_phi_'+str((i,k,l,omega)))
                    model.addGenConstrLog(phi[(i,k,l,omega)], log_phi[(i,k,l,omega)], name = 'log_phi_constr_'+str((i,k,l,omega)))

                for i in N_k_nod[k]:
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l,omega)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == 0, name = 'transshipment_flow_balance_'+str((i,k,l,omega)))

        if x_start:
            for k in K:
                for l in L_k[k]:
                    for i in N_k_d[k]:
                        model.getVarByName('phi_'+str((i,k,l,omega))).Start = x_star_EV['phi_'+str((i,k,l))]
                        model.update()

        if warm_start_sequential and prev_solution is not None:
            # Same feasible region every scenario (alpha only appears in the objective), so
            # the previous scenario's full optimal solution is a provably-feasible MIP start.
            for key, v in x_star.items():
                v.Start = prev_solution['x_star'][key]
            for key, v in w.items():
                v.Start = prev_solution['w'][key]
            for key, v in y.items():
                v.Start = prev_solution['y'][key]
            for key, v in delta_plus.items():
                v.Start = prev_solution['delta_plus'][key]
            for key, v in z_star.items():
                v.Start = prev_solution['z_star'][key]
            for (i,k,l) in prev_solution['phi']:
                phi[(i,k,l,omega)].Start = prev_solution['phi'][(i,k,l)]
                log_phi[(i,k,l,omega)].Start = prev_solution['log_phi'][(i,k,l)]
                delta_minus[(i,k,l,omega)].Start = prev_solution['delta_minus'][(i,k,l)]
            for (i,j,k,l) in prev_solution['x']:
                x[(i,j,k,l,omega)].Start = prev_solution['x'][(i,j,k,l)]
            for key, v in z.items():
                v.Start = prev_solution['z'][key]

        model.update()

        model.optimize()

        if diagnostics is not None:
            diagnostics.append(collect_solve_diagnostics(model, label='WS' + ('_warmstart' if warm_start_sequential else ''), scenario=omega))

        x_star_WS[omega] = {var.varName:var.x for var in model.getVars()}
        OF_omega_obj_val = sum(alpha[(i,k,l,omega)]*math.log(x_star_WS[omega]["phi_"+str((i,k,l,omega))]) if x_star_WS[omega]["phi_"+str((i,k,l,omega))] > 0 else alpha[(i,k,l,omega)]*math.log(1e-9) for k in K for l in L_k[k] for i in N_k_d[k])
        OF_omega.append(OF_omega_obj_val) 

        if warm_start_sequential:
            prev_solution = {
                'x_star': {key: v.X for key, v in x_star.items()},
                'w': {key: v.X for key, v in w.items()},
                'y': {key: v.X for key, v in y.items()},
                'delta_plus': {key: v.X for key, v in delta_plus.items()},
                'z_star': {key: v.X for key, v in z_star.items()},
                'phi': {(i,k,l): phi[(i,k,l,omega)].X for k in K for l in L_k[k] for i in N_k_d[k]},
                'log_phi': {(i,k,l): log_phi[(i,k,l,omega)].X for k in K for l in L_k[k] for i in N_k_d[k]},
                'delta_minus': {(i,k,l): delta_minus[(i,k,l,omega)].X for k in K for l in L_k[k] for i in N_k_d[k]},
                'x': {(i,j,k,l): x[(i,j,k,l,omega)].X for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])},
                'z': {key: v.X for key, v in z.items()},
            }


        del model

    results = {}

    results['proportional_fairness'] = [sum(alpha[(i,k,l,omega)]*math.log(x_star_WS[omega]['phi_'+str((i,k,l,omega))]) if x_star_WS[omega]['phi_'+str((i,k,l,omega))] > 0 else alpha[(i,k,l,omega)]*math.log(0.1e-9) for k in K for l in L_k[k] for i in N_k_d[k]) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['alpha_'+str((i,k,l))] = [alpha[(i,k,l,omega)] for omega in OMEGA]
                results['phi_'+str((i,k,l))] = [x_star_WS[omega]['phi_'+str((i,k,l,omega))] for omega in OMEGA]
                results['demand_'+str((i,k,l))] = [abs(b[(i,k,l)]) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['log_phi_'+str((i,k,l))] = [math.log(x_star_WS[omega]['phi_'+str((i,k,l,omega))]) if x_star_WS[omega]['phi_'+str((i,k,l,omega))] > 0 else math.log(0.1e-9) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                results['delta_plus_'+str((i,k,l))] = [x_star_WS[omega]['delta_plus_'+str((i,k,l))] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                results['supply_'+str((i,k,l))] = [b[(i,k,l)] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['delta_minus_'+str((i,k,l))] = [x_star_WS[omega]['delta_minus_'+str((i,k,l,omega))] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    results['x_'+str((i,j,k,l))] = [x_star_WS[omega]['x_'+str((i,j,k,l,omega))] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if i in N_k_s[k] and j in N_k[k]:
                    results['x_star_'+str((i,j,k,l))] = [x_star_WS[omega]['x_star_'+str((i,j,k,l))] for omega in OMEGA]

    for k in K:
        for i in N_k[k]:
            results['w_'+str((i,k))] = [x_star_WS[omega]['w_'+str((i,k))] for omega in OMEGA]

    for k in K:
        for (i,j) in A_k[k]:
            results['y_'+str((i,j,k))] = [x_star_WS[omega]['y_'+str((i,j,k))] for omega in OMEGA]

    save_results_csv(results, results_path(f'WS_scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{len(OMEGA)}_equal_probabilities_{True}_nat_log_linear_pieces_length_{pieces_length}{"_warmstart" if warm_start_sequential else ""}.csv'))

    OF_WS = sum(OF for OF in OF_omega)/len(OMEGA)

    return x_star_WS, OF_WS
