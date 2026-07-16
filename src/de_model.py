"""Deterministic Equivalent (DE) model: single model spanning all scenarios."""

import math
import os

import numpy as np
from gurobipy import GRB, Model, quicksum

from gurobi_utils import configure_common_gurobi_params, save_results_csv, collect_solve_diagnostics
from paths import results_path


def DE(b, OMEGA, N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, N_k_d, N_k_s, N_k_nod, N_k_d_nod, scarce, rho, x_start = {}, diagnostics=None):
    """`diagnostics`: optional list; if provided, a solve-diagnostics record
    for this (single, all-scenarios) DE solve is appended to it - see
    `run_experiments.write_solve_diagnostics_md`.
    """

    print(f'Deterministic equivalent')

    model = Model('stochastic_tdINDP')

    '''
    Gurobi parameters

    '''

    utility_type = 'power'
    pieces_length = 1
    configure_common_gurobi_params(model, output_flag=1)

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

    proportional_fairness = {omega:model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'proportional_fairness_'+str(omega)) for omega in OMEGA}
    phi = {(i,k,l,omega):model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub = abs(b[(i,k,l)]),name="phi_"+str((i,k,l,omega))) for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA}
    log_phi = {(i,k,l,omega):model.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,ub=math.log(abs(b[(i,k,l)])), name="log_phi_"+str((i,k,l,omega))) for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA}
    delta_minus = {(i,k,l,omega): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_minus_'+str((i,k,l,omega))) for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA}
    x = {(i,j,k,l,omega): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_'+str((i,j,k,l,omega))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k]) for omega in OMEGA}

    model.update()

    '''
    Objective functions:

    '''
    if utility_type == 'linear':
        # Linear
        for omega in OMEGA:
            model.addConstr(proportional_fairness[omega] == quicksum(log_phi[(i,k,l,omega)] + math.log(alpha[(i,k,l,omega)]) for k in K for l in L_k[k] for i in N_k_d[k]), name = 'proportional_fairness_constraint')
    else:
        # Power
        for omega in OMEGA:
            model.addConstr(proportional_fairness[omega] == quicksum(alpha[(i,k,l,omega)]*log_phi[(i,k,l,omega)] for k in K for l in L_k[k] for i in N_k_d[k]), name = 'proportional_fairness_constraint')

    model.setObjective(quicksum(rho[omega]*proportional_fairness[omega] for omega in OMEGA), GRB.MAXIMIZE)
    model.update()

    '''

    Constraints

    '''

    for k in K:
        for l in L_k[k]:

            for i in N_k_s[k]:
                for omega in OMEGA:
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
            for omega in OMEGA:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*y[(i,j,k)], name = 'arc_functionality_'+str((i,j,k,omega)))
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*w[(i,k)], name = 'node_i_functionality_'+str((i,j,k,omega)))
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*w[(j,k)], name = 'node_j_functionality_'+str((i,j,k,omega)))

    # Direction-exclusivity: every directed arc gets its own z (first-stage z_star for x_star
    # supply arcs - no omega, since x_star is a single first-stage decision; second-stage z for x
    # demand/transshipment arcs, per scenario); z_ij + z_ji <= 1 stops both directions of the same
    # pair from carrying flow at once.
    z_star = {(i,j,k,l): model.addVar(vtype = GRB.BINARY, name = 'z_star_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if i in N_k_s[k] and j in N_k[k]}
    z = {(i,j,k,l,omega): model.addVar(vtype = GRB.BINARY, name = 'z_'+str((i,j,k,l,omega))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k]) for omega in OMEGA}
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
                    for omega in OMEGA:
                        model.addConstr(x[(i,j,k,l,omega)] <= u[(i,j,k)]*z[(i,j,k,l,omega)], name = 'dir_excl_x_'+str((i,j,k,l,omega)))
                        if i != j and (j,i,k,l,omega) in z:
                            model.addConstr(z[(i,j,k,l,omega)] + z[(j,i,k,l,omega)] <= 1, name = 'dir_excl_x_pair_'+str((i,j,k,l,omega)))

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                for omega in OMEGA:
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l,omega)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == b[(i,k,l)] + delta_minus[(i,k,l,omega)], name = 'demand_flow_balance_'+str((i,k,l,omega)))
                    model.addConstr(phi[(i,k,l,omega)] == abs(b[(i,k,l)]) - delta_minus[(i,k,l,omega)], name = 'capture_phi_'+str((i,k,l,omega)))
                    model.addGenConstrLog(phi[(i,k,l,omega)], log_phi[(i,k,l,omega)], name = 'log_phi_constr_'+str((i,k,l,omega)))

            for i in N_k_nod[k]:
                for omega in OMEGA:
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l,omega)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == 0, name = 'transshipment_flow_balance_'+str((i,k,l,omega)))

    model.update()

    log_file_path = results_path(f'LOG_FILE_scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{len(OMEGA)}_equal_probabilities_{True}_nat_log_linear_pieces_length_{pieces_length}')
    if os.path.exists(log_file_path):
        # Gurobi appends to an existing LogFile rather than overwriting it, so remove any
        # stale log from a previous run to keep this file scoped to the current solve only.
        os.remove(log_file_path)
    model.params.LogFile = log_file_path
    model.optimize()

    if diagnostics is not None:
        diagnostics.append(collect_solve_diagnostics(model, label='DE'))

    proportional_fairness_values = [sum(alpha[(i,k,l,omega)]*math.log(model.getVarByName('phi_'+str((i,k,l,omega))).x) if model.getVarByName('phi_'+str((i,k,l,omega))).x > 0 else alpha[(i,k,l,omega)]*math.log(1e-9) for k in K for l in L_k[k] for i in N_k_d[k]) for omega in OMEGA]

    # Calculate mean and variance
    mean_value = np.mean(proportional_fairness_values)
    variance_value = np.var(proportional_fairness_values)
    std_value = np.std(proportional_fairness_values)

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                allocation = []
                for omega in OMEGA:
                    allocation.append(model.getVarByName("phi_"+str((i,k,l,omega))).x)
                # Calculate mean and variance
                mean_value = np.mean(allocation)
                variance_value = np.var(allocation)
                std_value = np.std(allocation)


    results = {}

    results['proportional_fairness'] = [sum(alpha[(i,k,l,omega)]*math.log(model.getVarByName('phi_'+str((i,k,l,omega))).x) if model.getVarByName('phi_'+str((i,k,l,omega))).x > 0 else alpha[(i,k,l,omega)]*math.log(0.1e-9) for k in K for l in L_k[k] for i in N_k_d[k]) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['alpha_'+str((i,k,l))] = [alpha[(i,k,l,omega)] for omega in OMEGA]
                results['phi_'+str((i,k,l))] = [model.getVarByName('phi_'+str((i,k,l,omega))).x for omega in OMEGA]
                results['demand_'+str((i,k,l))] = [abs(b[(i,k,l)]) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['log_phi_'+str((i,k,l))] = [math.log(model.getVarByName('phi_'+str((i,k,l,omega))).x) if model.getVarByName('phi_'+str((i,k,l,omega))).x > 0 else math.log(0.1e-9) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                results['delta_plus_'+str((i,k,l))] = [model.getVarByName('delta_plus_'+str((i,k,l))).x for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                results['supply_'+str((i,k,l))] = [b[(i,k,l)] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['delta_minus_'+str((i,k,l))] = [model.getVarByName('delta_minus_'+str((i,k,l,omega))).x for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    results['x_'+str((i,j,k,l))] = [model.getVarByName('x_'+str((i,j,k,l,omega))).x for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if i in N_k_s[k] and j in N_k[k]:
                    results['x_star_'+str((i,j,k,l))] = [model.getVarByName('x_star_'+str((i,j,k,l))).x for omega in OMEGA]

    for k in K:
        for i in N_k[k]:
            results['w_'+str((i,k))] = [model.getVarByName('w_'+str((i,k))).x for omega in OMEGA]

    for k in K:
        for (i,j) in A_k[k]:
            results['y_'+str((i,j,k))] = [model.getVarByName('y_'+str((i,j,k))).x for omega in OMEGA]

    save_results_csv(results, results_path(f'DE_scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{len(OMEGA)}_equal_probabilities_{True}_nat_log_linear_pieces_length_{pieces_length}.csv'))

    if len(OMEGA) == 1:
        x_start = {(i,k,l):model.getVarByName('phi_'+str((i,k,l,omega))).x for k in K for l in L_k[k] for i in N_k_d[k]}
        with open(results_path(f"utiltiy_{utility_type}_scarce_{scarce}_x_start.txt"), "w") as file:
            file.write(f'{x_start}')

    x_star_DE = {var.varName:var.x for var in model.getVars()}
    OF_DE = sum(rho[omega]*alpha[(i,k,l,omega)]*math.log(x_star_DE["phi_"+str((i,k,l,omega))]) if x_star_DE["phi_"+str((i,k,l,omega))] > 0 else rho[omega]*alpha[(i,k,l,omega)]*math.log(1e-9) for k in K for l in L_k[k] for i in N_k_d[k] for omega in OMEGA)
    return x_star_DE, OF_DE
