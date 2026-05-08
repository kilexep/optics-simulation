import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_legacy_pet_water_stl_optical_smoke_demo.py"
)


def test_demo_runs_against_synthetic_solid_cylinder_stl(
    tmp_path: Path,
) -> None:
    assert DEMO_SCRIPT.is_file(), f"demo script not found at {DEMO_SCRIPT}"

    # Build a synthetic SOLID cylinder STL inside tmp_path so the
    # test does not depend on the user's real PET-bottle STL. A
    # solid cylinder has consistent outward vertex normals, which
    # lets auto inner-offset mode strictly reduce the median
    # radial distance and so satisfy the auto-mode invariant
    # (inward_offset_detected == True). A closed double-walled
    # synthetic shell ties on the median and is intentionally
    # avoided here.
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
            "--sample-count", "16",
            "--inner-offset-mode", "auto",
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
    assert "Legacy PET-water STL optical smoke demo" in result.stdout
    assert "Legacy-style STL optical smoke check" in result.stdout
    assert (
        "Legacy STL inner-offset orientation diagnostic"
        in result.stdout
    )
    assert "Target height:" in result.stdout
    assert "Target diameter:" in result.stdout
    assert "Wall thickness:" in result.stdout
    assert "Scale xyz:" in result.stdout
    assert "Inner offset mode:" in result.stdout
    assert "Selected offset sign:" in result.stdout
    assert "Original radial stat:" in result.stdout
    assert "Selected radial stat:" in result.stdout
    assert "Inward offset detected:" in result.stdout
    assert "Step count:" in result.stdout
    assert "Final rays:" in result.stdout
    assert "Detector hits:" in result.stdout
    assert "Max relative irradiance:" in result.stdout
    assert "Invariants: PASS" in result.stdout
