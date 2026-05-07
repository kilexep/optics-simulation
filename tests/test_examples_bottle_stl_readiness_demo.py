import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_bottle_stl_readiness_demo.py"
)


def test_demo_script_runs_in_synthetic_fallback_mode() -> None:
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
    assert "Bottle STL readiness demo" in result.stdout
    assert "Bottle STL readiness runner" in result.stdout
    assert "External STL supplied:" in result.stdout
    assert "Mesh path:" in result.stdout
    assert "Load success:" in result.stdout
    assert "Scale factor:" in result.stdout
    assert "Vertex count:" in result.stdout
    assert "Face count:" in result.stdout
    assert "Estimated radius:" in result.stdout
    assert "Fit quality:" in result.stdout
    assert "Simulation ready:" in result.stdout
    assert "Blocking issues:" in result.stdout
    assert "Warnings:" in result.stdout
    assert "Invariants: PASS" in result.stdout
