"""Expected Value (EV) and Expectation of the Expected Value Solution (EEVS) model."""

import math

from gurobipy import GRB, Model, quicksum

from gurobi_utils import configure_common_gurobi_params, save_results_csv, collect_solve_diagnostics
from paths import results_path


def EV_and_EEVS(alpha_mu_type, b, OMEGA, N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, N_k_d, N_k_s, N_k_nod, N_k_d_nod, scarce, rho, diagnostics=None):
    '''
    `diagnostics`: optional list; if provided, one `collect_solve_diagnostics(...)`
    record per Gurobi solve (the EV solve, then one per EEVS scenario) is
    appended to it in place, for audit/reproducibility reporting - see
    `run_experiments.write_solve_diagnostics_md`.

    Expected value

    '''

    print('Expected value')
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

    proportional_fairness = model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'proportional_fairness')
    phi = {(i,k,l):model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub = abs(b[(i,k,l)]),name="phi_"+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_d[k]}
    log_phi = {(i,k,l):model.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,ub=math.log(abs(b[(i,k,l)])), name="log_phi_"+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_d[k]}
    delta_minus = {(i,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_minus_'+str((i,k,l))) for k in K for l in L_k[k] for i in N_k_d[k]}
    x = {(i,j,k,l): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_'+str((i,j,k,l))) for k in K for l in L_k[k] for (i,j) in A_k[k] if (i in N_k_d_nod[k] and j in N_k_d_nod[k])}

    model.update()

    '''
    Objective functions:

    '''
    if utility_type == 'linear':
        # Linear
        model.addConstr(proportional_fairness == quicksum(log_phi[(i,k,l)] + math.log(0.5) for k in K for l in L_k[k] for i in N_k_d[k]), name = 'proportional_fairness_constraint')
    else:
        # Power
        model.addConstr(proportional_fairness == quicksum(alpha_mu_type[(i,k,l)]*log_phi[(i,k,l)] for k in K for l in L_k[k] for i in N_k_d[k]), name = 'proportional_fairness_constraint')

    model.setObjective(proportional_fairness, GRB.MAXIMIZE)
    model.update()

    '''

    Constraints

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
                if (i,j,k,k_tilde) in gamma.keys():
                    model.addConstr(quicksum(w[(i,k)]*gamma[(i,j,k,k_tilde)] for i in N_k[k]) <= w[(j,k_tilde)], name = 'interdependent_functionality_'+str((k,k_tilde,j)))



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
                model.addGenConstrLog(phi[(i,k,l)], log_phi[(i,k,l)], name = 'log_phi_constr_'+str((i,k,l)))

            for i in N_k_nod[k]:
                model.addConstr(quicksum(x[(i,j,k,l)] for j in N_k_d_nod[k] if (i,j) in A_k[k]) - (quicksum(x[(j,i,k,l)] for j in N_k_d_nod[k] if (j,i) in A_k[k]) + quicksum(x_star[(j,i,k,l)] for j in N_k_s[k] if (j,i) in A_k[k])) == 0, name = 'transshipment_flow_balance_'+str((i,k,l)))

    model.update()

    model.optimize()

    if diagnostics is not None:
        diagnostics.append(collect_solve_diagnostics(model, label='EV'))

    x_star_EV = {var.varName:var.x for var in model.getVars()}
    # OF_EV = model.objVal # to be computed with the natural log
    OF_EV = sum(alpha_mu_type[(i,k,l)]*math.log(x_star_EV["phi_"+str((i,k,l))]) if x_star_EV["phi_"+str((i,k,l))] > 0 else alpha_mu_type[(i,k,l)]*math.log(1e-9) for k in K for l in L_k[k] for i in N_k_d[k])

    del model

    ''' 

    Expectation of the expected value

    '''

    x_star_EEVS = {}
    OF_omega = []
    for omega in OMEGA:
        print(f'Expectation of the expected value, scenario {omega+1} out of {len(OMEGA)}')

        model = Model('stochastic_tdINDP')

        '''
        Gurobi parameters

        '''
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
                    if (i,j,k,k_tilde) in gamma.keys():
                        model.addConstr(quicksum(w[(i,k)]*gamma[(i,j,k,k_tilde)] for i in N_k[k]) <= w[(j,k_tilde)], name = 'interdependent_functionality_'+str((k,k_tilde,j)))



        for k in K:
            for (i,j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*y[(i,j,k)], name = 'arc_functionality_'+str((i,j,k,omega)))
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*w[(i,k)], name = 'node_i_functionality_'+str((i,j,k,omega)))
                    model.addConstr(quicksum(x[(i,j,k,l,omega)] for l in L_k[k]) <= u[(i,j,k)]*w[(j,k)], name = 'node_j_functionality_'+str((i,j,k,omega)))

        # Direction-exclusivity: every directed arc gets its own z (first-stage z_star for
        # x_star supply arcs, second-stage z for x demand/transshipment arcs); z_ij + z_ji <= 1
        # stops both directions of the same pair from carrying flow at once.
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

        # EEVS fixes ALL of the true first-stage decisions to the EV solution, not just
        # x_star: w/y (topology) and z_star (direction-exclusivity binaries) have no
        # per-scenario recourse in this model either, since alpha only ever appears in the
        # objective, never in a constraint. Fixing x_star alone while leaving w/y/z_star free
        # forces the solver to re-derive a fresh combination of binaries consistent with the
        # exact fixed continuous values, which can spuriously come back infeasible even
        # though the EV solution itself is already a feasible replay. Fixing everything
        # first-stage sidesteps that and matches the standard EEVS definition (fix
        # first-stage decisions, optimize recourse given the scenario).
        #
        # Continuous first-stage values (x_star, delta_plus) are fixed within a small
        # tolerance band rather than via exact equality: Gurobi's own EV solve can return
        # values that sit a few 1e-9 outside a tight capacity bound (ordinary LP solve
        # noise, well within its default feasibility tolerance) -- an exact `==` re-applies
        # that tiny residual as a hard equality on top of the *same* capacity constraint in
        # a fresh model, which can tip a razor-thin-feasible arc into "infeasible". A 1e-6
        # band absorbs that noise without materially changing what "fixed to the EV
        # solution" means. Binaries (w, y, z_star) are 0/1-valued with no such noise, so
        # they are rounded and fixed exactly.
        fix_tol = 1e-6
        for k in K:
            for (i,j) in A_k[k]:
                for l in L_k[k]:
                    if i in N_k_s[k] and j in N_k[k]:
                        ev_val = x_star_EV['x_star_'+str((i,j,k,l))]
                        model.addConstr(x_star[(i,j,k,l)] <= ev_val + fix_tol, name = 'fix_first_stage_x_star_ub_'+str((i,j,k,l)))
                        model.addConstr(x_star[(i,j,k,l)] >= ev_val - fix_tol, name = 'fix_first_stage_x_star_lb_'+str((i,j,k,l)))
                        model.addConstr(z_star[(i,j,k,l)] == round(x_star_EV['z_star_'+str((i,j,k,l))]), name = 'fix_first_stage_z_star_'+str((i,j,k,l)))
        for k in K:
            for i in N_k[k]:
                model.addConstr(w[(i,k)] == round(x_star_EV['w_'+str((i,k))]), name = 'fix_first_stage_w_'+str((i,k)))
            for (i,j) in A_k[k]:
                model.addConstr(y[(i,j,k)] == round(x_star_EV['y_'+str((i,j,k))]), name = 'fix_first_stage_y_'+str((i,j,k)))
            for l in L_k[k]:
                for i in N_k_s[k]:
                    ev_val = x_star_EV['delta_plus_'+str((i,k,l))]
                    model.addConstr(delta_plus[(i,k,l)] <= ev_val + fix_tol, name = 'fix_first_stage_delta_plus_ub_'+str((i,k,l)))
                    model.addConstr(delta_plus[(i,k,l)] >= ev_val - fix_tol, name = 'fix_first_stage_delta_plus_lb_'+str((i,k,l)))

        model.update()

        model.optimize()

        if diagnostics is not None:
            diagnostics.append(collect_solve_diagnostics(model, label='EEVS', scenario=omega))

        if model.SolCount == 0:
            raise RuntimeError(
                f"EEVS scenario {omega} (status {model.status}) produced no feasible "
                "solution when re-solved with the EV solution's first-stage decisions "
                "fixed. This should not happen since the EV solution itself is a feasible "
                "replay for any scenario (alpha only affects the objective, not "
                "constraints) -- check for numerical tolerance issues or a modeling bug."
            )

        x_star_EEVS[omega] = {var.varName:var.x for var in model.getVars()}
        OF_omega_obj_val = sum(alpha[(i,k,l,omega)]*math.log(x_star_EEVS[omega]["phi_"+str((i,k,l,omega))]) if x_star_EEVS[omega]["phi_"+str((i,k,l,omega))] > 0 else alpha[(i,k,l,omega)]*math.log(1e-9) for k in K for l in L_k[k] for i in N_k_d[k])
        OF_omega.append(OF_omega_obj_val)

        del model

    results = {}

    results['proportional_fairness'] = [sum(alpha[(i,k,l,omega)]*math.log(x_star_EEVS[omega]['phi_'+str((i,k,l,omega))]) if x_star_EEVS[omega]['phi_'+str((i,k,l,omega))] > 0 else alpha[(i,k,l,omega)]*math.log(0.1e-9) for k in K for l in L_k[k] for i in N_k_d[k]) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['alpha_'+str((i,k,l))] = [alpha[(i,k,l,omega)] for omega in OMEGA]
                results['phi_'+str((i,k,l))] = [x_star_EEVS[omega]['phi_'+str((i,k,l,omega))] for omega in OMEGA]
                results['demand_'+str((i,k,l))] = [abs(b[(i,k,l)]) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['log_phi_'+str((i,k,l))] = [math.log(x_star_EEVS[omega]['phi_'+str((i,k,l,omega))]) if x_star_EEVS[omega]['phi_'+str((i,k,l,omega))] > 0 else math.log(0.1e-9) for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                results['delta_plus_'+str((i,k,l))] = [x_star_EEVS[omega]['delta_plus_'+str((i,k,l))] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_s[k]:
                results['supply_'+str((i,k,l))] = [b[(i,k,l)] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for i in N_k_d[k]:
                results['delta_minus_'+str((i,k,l))] = [x_star_EEVS[omega]['delta_minus_'+str((i,k,l,omega))] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if (i in N_k_d_nod[k] and j in N_k_d_nod[k]):
                    results['x_'+str((i,j,k,l))] = [x_star_EEVS[omega]['x_'+str((i,j,k,l,omega))] for omega in OMEGA]

    for k in K:
        for l in L_k[k]:
            for (i,j) in A_k[k]:
                if i in N_k_s[k] and j in N_k[k]:
                    results['x_star_'+str((i,j,k,l))] = [x_star_EEVS[omega]['x_star_'+str((i,j,k,l))] for omega in OMEGA]

    for k in K:
        for i in N_k[k]:
            results['w_'+str((i,k))] = [x_star_EEVS[omega]['w_'+str((i,k))] for omega in OMEGA]

    for k in K:
        for (i,j) in A_k[k]:
            results['y_'+str((i,j,k))] = [x_star_EEVS[omega]['y_'+str((i,j,k))] for omega in OMEGA]

    save_results_csv(results, results_path(f'EEVS_mu_hat_scarce_{scarce}_utility_type_{utility_type}_num_scenarios_{len(OMEGA)}_equal_probabilities_{True}_nat_log_linear_pieces_length_{pieces_length}.csv'))

    OF_EEVS = sum(OF for OF in OF_omega)/len(OMEGA)

    return x_star_EV, OF_EV, x_star_EEVS, OF_EEVS
