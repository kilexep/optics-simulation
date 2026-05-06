import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT / "examples" / "run_metric_convergence_diagnostic_demo.py"
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
        timeout=180,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Metric convergence diagnostic demo" in result.stdout
    assert "Synthetic metric-convergence smoke check" in result.stdout
    assert "Detector resolutions:" in result.stdout
    assert "Ray grid sizes:" in result.stdout
    assert "Top-percent values:" in result.stdout
    assert "Resolution" in result.stdout
    assert "Ray grid" in result.stdout
    assert "Top percent" in result.stdout
    assert "C99" in result.stdout
    assert "Cmax" in result.stdout
    assert "Detector hits" in result.stdout
    assert "Quantization warning" in result.stdout
    assert "Invariants: PASS" in result.stdout
