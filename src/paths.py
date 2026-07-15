"""Central path definitions so model outputs always land in a predictable place
regardless of the current working directory the notebook/script is launched from.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def results_path(filename):
    """Return the absolute path for `filename` inside the results/ directory."""
    return str(RESULTS_DIR / filename)
