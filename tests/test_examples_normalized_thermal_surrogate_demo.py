import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT / "examples" / "run_normalized_thermal_surrogate_demo.py"
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
    assert "Normalized thermal surrogate demo" in result.stdout
    assert (
        "Synthetic normalized optical-to-thermal surrogate smoke check"
        in result.stdout
    )
    assert "Max relative irradiance:" in result.stdout
    assert "Thermal incident flux surrogate:" in result.stdout
    assert "Absorbed flux surrogate:" in result.stdout
    assert "Max temperature:" in result.stdout
    assert "Final temperature:" in result.stdout
    assert "Time to threshold:" in result.stdout
    assert "Invariants: PASS" in result.stdout
