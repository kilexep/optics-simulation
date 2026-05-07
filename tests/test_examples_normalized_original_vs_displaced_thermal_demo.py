import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_normalized_original_vs_displaced_thermal_demo.py"
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
    assert "Normalized original-vs-displaced thermal demo" in result.stdout
    assert (
        "Synthetic normalized optical-to-thermal comparison smoke check"
        in result.stdout
    )
    assert "Mode: inward" in result.stdout
    assert "Moved vertex count:" in result.stdout
    assert "Original max relative irradiance:" in result.stdout
    assert "Displaced max relative irradiance:" in result.stdout
    assert "Delta max relative irradiance:" in result.stdout
    assert "Original max temperature:" in result.stdout
    assert "Displaced max temperature:" in result.stdout
    assert "Delta max temperature:" in result.stdout
    assert "Original time to threshold:" in result.stdout
    assert "Displaced time to threshold:" in result.stdout
    assert "Mesh vertices mutated: False" in result.stdout
    assert "Mesh faces mutated: False" in result.stdout
    assert "Invariants: PASS" in result.stdout
