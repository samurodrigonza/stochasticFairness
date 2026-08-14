"""Central path definitions so model outputs always land in a predictable place
regardless of the current working directory the notebook/script is launched from.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
SOLVER_LOGS_DIR = RESULTS_DIR / "solver_logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def results_path(filename):
    """Return the absolute path for `filename` inside the results/ directory."""
    return str(RESULTS_DIR / filename)


def solver_log_path(filename):
    """Return the absolute path for a per-solve Gurobi log inside
    results/solver_logs/ (created on demand). Kept in a dedicated subfolder so
    the many per-scenario log files (EV/EEVS/WS/DE) don't clutter results/.
    """
    SOLVER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(SOLVER_LOGS_DIR / filename)
