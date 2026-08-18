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

#: Default relative MIP gap for every model in the pipeline.
#:
#: Override for a whole run without editing code by setting the environment
#: variable ``STOCHFAIR_MIPGAP`` before importing this module, e.g. from a
#: notebook cell::
#:
#:     import os; os.environ['STOCHFAIR_MIPGAP'] = '1e-6'
#:
#: The value actually used is recorded per model in ``summary.json``'s
#: ``solver_config`` block (see :func:`solver_config_snapshot`), so a run
#: always documents its own gap regardless of how it was set.
DEFAULT_MIP_GAP = float(os.environ.get('STOCHFAIR_MIPGAP', 1e-4))


def configure_common_gurobi_params(model, num_focus=0, output_flag=1, log_file=None, func_nonlinear=1, mip_gap=None,
                                   integrality_focus=0, int_feas_tol=1e-7):
    """Apply the Gurobi parameter settings shared by all tdINDP models.

    If `log_file` is given, the full Gurobi solver log for this solve is written
    there. Any existing file at that path is removed first, because Gurobi
    appends to (rather than overwrites) an existing LogFile across runs -- so
    each per-solve log stays scoped to just the current solve.

    `func_nonlinear` selects how the ln(phi) general constraints are handled.
    All models now use 1 (dynamic outer approximation), which keeps every model
    small -- the static piecewise-linear expansion otherwise adds ~20k columns
    and an SOS2 set per solve, and was the cause of the EEVS/WS solves grinding
    for hours.

    `mip_gap` is `DEFAULT_MIP_GAP` (1e-4 unless the `STOCHFAIR_MIPGAP`
    environment variable overrides it), matching Gurobi's own default but set
    explicitly so the value is recorded in each run's solver-config snapshot
    rather than being implicit.

    The numerical settings were previously NumericFocus=3, IntegralityFocus=1
    and IntFeasTol=1e-9. That combination demanded integrality feasibility six
    orders of magnitude tighter than the accuracy of the model's own nonlinear
    constraints: under the piecewise-linear expansion every solve reported
    `max constraint violation ~5e-04 exceeds tolerance`. With func_nonlinear=1
    the ln(phi) constraints are handled exactly, so that conflict is gone and
    the conservative (expensive) numerics are no longer justified. IntFeasTol
    is still 100x tighter than Gurobi's 1e-5 default.
    """
    model.setParam('OutputFlag', output_flag)
    model.setParam('FuncNonlinear', func_nonlinear)
    model.setParam('FuncPieceRatio', -1)
    model.setParam('FuncPieceError', 1e-4)  # only consulted when func_nonlinear=0
    model.setParam('DualReductions', 0)
    model.setParam('NumericFocus', num_focus)
    model.setParam('IntegralityFocus', integrality_focus)
    model.setParam('IntFeasTol', int_feas_tol)
    if mip_gap is None:
        mip_gap = DEFAULT_MIP_GAP
    if mip_gap is not None:
        model.setParam('MIPGap', mip_gap)
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
        'solver_config': solver_config_snapshot(model),
        'flags': ', '.join(flags) if flags else '',
    }


def solver_config_snapshot(model):
    """The solver settings this model was actually solved under.

    Recorded per solve so a results file on disk is self-describing: the
    ln(phi) treatment (`FuncNonlinear`) and the numerical tolerances have both
    changed during this project and produce materially different runtimes, so
    "which configuration produced this number" must not depend on memory or on
    what happens to be checked into `gurobi_utils` today.
    """
    names = ('FuncNonlinear', 'FuncPieceError', 'NumericFocus',
             'IntegralityFocus', 'IntFeasTol', 'MIPGap')
    snapshot = {}
    for name in names:
        try:
            snapshot[name] = model.getParamInfo(name)[2]  # current value
        except Exception:
            snapshot[name] = None
    return snapshot


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
