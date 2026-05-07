import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_thermal_risk_metrics_original_vs_displaced_demo.py"
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
    assert "Thermal risk metrics original-vs-displaced demo" in result.stdout
    assert "Synthetic thermal-risk surrogate metrics" in result.stdout
    assert "not an ignition validation model" in result.stdout
    assert "Mode: inward" in result.stdout
    assert "Original max temperature:" in result.stdout
    assert "Displaced max temperature:" in result.stdout
    assert "Delta max temperature:" in result.stdout
    assert "Original top-percent temperature rise:" in result.stdout
    assert "Displaced top-percent temperature rise:" in result.stdout
    assert "Original threshold exceeded count:" in result.stdout
    assert "Displaced threshold exceeded count:" in result.stdout
    assert "Delta threshold exceeded count:" in result.stdout
    assert "Original threshold exceeded fraction:" in result.stdout
    assert "Displaced threshold exceeded fraction:" in result.stdout
    assert "Mesh vertices mutated: False" in result.stdout
    assert "Mesh faces mutated: False" in result.stdout
    assert "Invariants: PASS" in result.stdout
