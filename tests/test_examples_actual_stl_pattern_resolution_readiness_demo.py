import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_actual_stl_pattern_resolution_readiness_demo.py"
)


def test_demo_runs_against_synthetic_solid_cylinder_stl(
    tmp_path: Path,
) -> None:
    assert DEMO_SCRIPT.is_file(), f"demo script not found at {DEMO_SCRIPT}"

    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from optics_simulation.geometry import (
            create_subdivided_synthetic_bottle_body,
        )
    finally:
        sys.path.pop(0)

    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=64, height_segments=12,
    )
    stl_path = tmp_path / "synthetic_solid_cylinder.stl"
    mesh.export(stl_path)

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path + os.pathsep + existing if existing else src_path
    )

    result = subprocess.run(
        [
            sys.executable,
            str(DEMO_SCRIPT),
            "--mesh", str(stl_path),
            "--target-height", "225.6",
            "--target-diameter", "72.1",
            "--wall-thickness", "0.3",
            "--sigma", "0.03",
            "--max-depth", "0.05",
            "--subdivision-iterations", "1",
        ],
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
    assert (
        "Actual STL pattern-resolution readiness demo"
        in result.stdout
    )
    assert (
        "Actual STL pattern-resolution readiness diagnostic"
        in result.stdout
    )
    assert "Original selected vertices:" in result.stdout
    assert "Original readiness:" in result.stdout
    assert "Subdivided selected vertices:" in result.stdout
    assert "Subdivided readiness:" in result.stdout
    assert "Subdivision iterations:" in result.stdout
    assert "Invariants: PASS" in result.stdout
