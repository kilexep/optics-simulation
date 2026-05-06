import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT / "examples" / "run_displaced_mesh_quality_demo.py"
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
    assert "Displaced mesh quality demo" in result.stdout
    assert "Synthetic displaced-mesh quality smoke check" in result.stdout
    assert "Mode: inward" in result.stdout
    assert "Original vertex count:" in result.stdout
    assert "Displaced vertex count:" in result.stdout
    assert "Original face count:" in result.stdout
    assert "Displaced face count:" in result.stdout
    assert "Moved vertex count:" in result.stdout
    assert "Max displacement:" in result.stdout
    assert "Mean displacement:" in result.stdout
    assert "Original watertight:" in result.stdout
    assert "Displaced watertight:" in result.stdout
    assert "Original bounds z:" in result.stdout
    assert "Displaced bounds z:" in result.stdout
    assert "Mesh vertices mutated: False" in result.stdout
    assert "Mesh faces mutated: False" in result.stdout
    assert "Invariants: PASS" in result.stdout
