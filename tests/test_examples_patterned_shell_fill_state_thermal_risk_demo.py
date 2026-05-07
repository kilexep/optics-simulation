import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_patterned_shell_fill_state_thermal_risk_demo.py"
)


def test_demo_script_runs_and_invariants_pass() -> None:
    assert DEMO_SCRIPT.is_file(), f"demo script not found at {DEMO_SCRIPT}"

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path + os.pathsep + existing if existing else src_path
    )

    result = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (
        "Patterned shell fill-state thermal risk demo" in result.stdout
    )
    assert (
        "Synthetic outer-surface patterned shell smoke check"
        in result.stdout
    )
    assert "Pattern applied to outer shell surface only" in result.stdout
    assert "Inner wall displaced vertices:" in result.stdout
    assert "Rim displaced vertices:" in result.stdout
    assert "Moved vertex count:" in result.stdout
    assert "Fill medium: air" in result.stdout
    assert "Fill medium: water" in result.stdout
    assert "Original shell max temperature:" in result.stdout
    assert "Patterned shell max temperature:" in result.stdout
    assert "Delta max temperature:" in result.stdout
    assert "Original threshold exceeded count:" in result.stdout
    assert "Patterned threshold exceeded count:" in result.stdout
    assert "Delta threshold exceeded count:" in result.stdout
    assert "Mesh vertices mutated: False" in result.stdout
    assert "Mesh faces mutated: False" in result.stdout
    assert "Medium tracking validation passed:" in result.stdout
    assert "Unexpected surface count:" in result.stdout
    assert "Trace epsilon:" in result.stdout
    assert "Validated side y range:" in result.stdout
    assert "Validated side z range:" in result.stdout
    assert "Invariants: PASS" in result.stdout
