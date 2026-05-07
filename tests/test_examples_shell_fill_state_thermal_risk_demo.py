import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_shell_fill_state_thermal_risk_demo.py"
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
    assert "Shell fill-state thermal risk demo" in result.stdout
    assert (
        "Synthetic shell/fill-state optical-to-thermal smoke check"
        in result.stdout
    )
    assert "Side-incidence ray grid" in result.stdout
    assert "Solid cylinder" in result.stdout
    assert "Empty shell" in result.stdout
    assert "Water-filled shell" in result.stdout
    assert "Solid max temperature:" in result.stdout
    assert "Empty shell max temperature:" in result.stdout
    assert "Water shell max temperature:" in result.stdout
    assert "Solid threshold exceeded count:" in result.stdout
    assert "Empty shell threshold exceeded count:" in result.stdout
    assert "Water shell threshold exceeded count:" in result.stdout
    assert "Invariants: PASS" in result.stdout
