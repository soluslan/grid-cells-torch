import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENTRY_POINTS = ["generate_data.py", "train.py", "analyze_supervised.py", "train_rl_agent.py",
                "eval_rl_agent.py"]


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_help_runs(script):
    if script == "eval_rl_agent.py":
        pytest.importorskip("deepmind_lab")
    subprocess.run([sys.executable, script, "--help"], cwd=REPO, check=True, capture_output=True)
