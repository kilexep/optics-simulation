import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT / "examples" / "run_vertex_displacement_amount_demo.py"
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
    assert "Vertex displacement amount demo" in result.stdout
    assert "Synthetic vertex-displacement-amount smoke check" in result.stdout
    assert "Vertex count:" in result.stdout
    assert "Active vertex count:" in result.stdout
    assert "Normalized depth max:" in result.stdout
    assert "Physical depth max:" in result.stdout
    assert "Mesh vertices mutated: False" in result.stdout
    assert "Mesh faces mutated: False" in result.stdout
    assert "Invariants: PASS" in result.stdout
