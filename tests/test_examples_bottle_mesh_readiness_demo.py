import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_bottle_mesh_readiness_demo.py"
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
    assert "Bottle mesh readiness demo" in result.stdout
    assert (
        "Synthetic/loaded bottle mesh readiness diagnostic"
        in result.stdout
    )
    assert "Simple solid cylinder" in result.stdout
    assert "Subdivided solid cylinder" in result.stdout
    assert "Synthetic shell" in result.stdout
    assert "Estimated radius:" in result.stdout
    assert "Radius CV:" in result.stdout
    assert "Fit quality:" in result.stdout
    assert "Simulation ready:" in result.stdout
    assert "Blocking issues:" in result.stdout
    assert "Warnings:" in result.stdout
    assert "Invariants: PASS" in result.stdout
