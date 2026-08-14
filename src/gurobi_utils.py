"""Shared Gurobi configuration and results post-processing helpers.

These utilities are reused by the EV/EEVS, WS (wait-and-see) and DE
(deterministic equivalent) model builders so that solver settings and the
summary-statistics/CSV export logic are defined in a single place.
"""

import os

import numpy as np
import pandas as pd

_STATUS_NAMES = {
    2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
    9: 'TIME_LIMIT', 13: 'SUBOPTIMAL', 12: 'NUMERIC', 6: 'CUTOFF',
    7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 10: 'SOLUTION_LIMIT',
    11: 'INTERRUPTED', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT',
}


def configure_common_gurobi_params(model, num_focus=3, output_flag=1, log_file=None):
    """Apply the Gurobi parameter settings shared by all tdINDP models.

    If `log_file` is given, the full Gurobi solver log for this solve is written
    there. Any existing file at that path is removed first, because Gurobi
    appends to (rather than overwrites) an existing LogFile across runs -- so
    each per-solve log stays scoped to just the current solve.
    """
    model.setParam('OutputFlag', output_flag)
    model.setParam('FuncPieceRatio', -1)
    model.setParam('FuncPieceError', 1e-4)  # tightens the piecewise-linear approximation of ln(phi) beyond the default
    model.setParam('DualReductions', 0)
    model.setParam('NumericFocus', num_focus)
    model.setParam('IntegralityFocus', 1)
    model.setParam('IntFeasTol', 1e-9)
    if log_file is not None:
        if os.path.exists(log_file):
            os.remove(log_file)
        model.setParam('LogFile', log_file)


def collect_solve_diagnostics(model, label, scenario=None):
    """Read back Gurobi's own solve diagnostics right after `model.optimize()`.

    Used for audit/reproducibility: every model solved anywhere in the
    Formulation A pipeline (EV, per-scenario EEVS, per-scenario WS, DE) should
    call this immediately after `model.optimize()` so a run leaves behind a
    persisted record of how each individual Gurobi solve actually terminated,
    not just the final headline objective value. See
    `run_experiments.write_solve_diagnostics_md`.

    `scenario` is an optional per-omega index (None for models with no
    per-scenario loop, e.g. EV or DE).
    """
    status = model.status
    sol_count = model.SolCount
    runtime = model.Runtime

    obj_val = obj_bound = mip_gap = None
    if sol_count > 0:
        try:
            obj_val = model.ObjVal
        except AttributeError:
            obj_val = None
    try:
        obj_bound = model.ObjBound
    except AttributeError:
        obj_bound = None
    try:
        mip_gap = model.MIPGap
    except AttributeError:
        mip_gap = None

    try:
        work_units = model.Work
    except AttributeError:
        work_units = None

    flags = []
    if sol_count == 0:
        flags.append('NO INCUMBENT')
    if status not in (2,):  # not GRB.OPTIMAL
        flags.append(_STATUS_NAMES.get(status, f'STATUS_{status}'))
    if mip_gap is not None and sol_count > 0 and mip_gap > 0.01 + 1e-9:
        flags.append(f'gap {mip_gap:.2%} > 1%')

    return {
        'label': label,
        'scenario': scenario,
        'status': status,
        'status_name': _STATUS_NAMES.get(status, f'STATUS_{status}'),
        'runtime_s': runtime,
        'sol_count': sol_count,
        'obj_val': obj_val,
        'obj_bound': obj_bound,
        'mip_gap': mip_gap,
        'num_vars': model.NumVars,
        'num_bin_vars': model.NumBinVars,
        'num_constrs': model.NumConstrs,
        'work_units': work_units,
        'flags': ', '.join(flags) if flags else '',
    }


def compute_row_statistics(row):
    """Row-wise summary statistics used to augment the results DataFrame."""
    return pd.Series({
        'mean': np.mean(row),
        'std': np.std(row),
        'min': np.min(row),
        'max': np.max(row),
        'median': np.median(row),
        'mode': row.mode().iloc[0],
        'q1': np.percentile(row, 25),
        'q2': np.percentile(row, 50),
        'q3': np.percentile(row, 75)
    })


def build_results_dataframe(results):
    """Transpose a {column_name: [values...]} dict into a DataFrame with summary stats appended as columns."""
    df = pd.DataFrame(results, columns=results.keys()).transpose()
    summary_stats = df.apply(compute_row_statistics, axis=1)
    return pd.concat([df, summary_stats], axis=1)


def save_results_csv(results, filename):
    """Build the summary-stats DataFrame for `results` and save it to `filename`."""
    df = build_results_dataframe(results)
    df.to_csv(filename)
    return df
