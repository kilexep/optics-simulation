import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT / "examples" / "run_angle_distance_sweep_demo.py"
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
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Angle-distance sweep demo" in result.stdout
    assert (
        "Synthetic angle-distance optical sweep smoke check"
        in result.stdout
    )
    assert "Angles:" in result.stdout
    assert "Detector z values:" in result.stdout
    assert "Result count:" in result.stdout
    assert "Max C99 angle:" in result.stdout
    assert "Max C99 detector z:" in result.stdout
    assert "Max C99:" in result.stdout
    assert "Angle 0.0" in result.stdout
    assert "C99:" in result.stdout
    assert "Cmax:" in result.stdout
    assert "Detector hits:" in result.stdout
    assert "Termination:" in result.stdout
    assert "Invariants: PASS" in result.stdout
