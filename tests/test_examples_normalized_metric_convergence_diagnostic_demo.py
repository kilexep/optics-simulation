import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_normalized_metric_convergence_diagnostic_demo.py"
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
        timeout=240,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Normalized metric convergence diagnostic demo" in result.stdout
    assert (
        "Synthetic normalized metric-convergence smoke check"
        in result.stdout
    )
    assert "use_relative_irradiance=True" in result.stdout
    assert "use_power_weights=True" in result.stdout
    assert "Detector resolutions:" in result.stdout
    assert "Ray grid sizes:" in result.stdout
    assert "Top-percent values:" in result.stdout
    assert "Result count:" in result.stdout
    assert "Sweep A" in result.stdout
    assert "Sweep B" in result.stdout
    assert "Sweep C" in result.stdout
    assert "Raw weighted C99" in result.stdout
    assert "Normalized weighted C99" in result.stdout
    assert "Delta normalized-vs-raw C99" in result.stdout
    assert "Raw weighted Cmax" in result.stdout
    assert "Normalized weighted Cmax" in result.stdout
    assert "Delta normalized-vs-raw Cmax" in result.stdout
    assert "Detector hits" in result.stdout
    assert "Final rays" in result.stdout
    assert "Normalized saturation count:" in result.stdout
    assert "Quantization warning" in result.stdout
    assert "Invariants: PASS" in result.stdout
