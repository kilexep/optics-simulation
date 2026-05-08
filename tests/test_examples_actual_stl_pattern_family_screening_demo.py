import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_actual_stl_pattern_family_screening_demo.py"
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
            "--inner-offset-mode", "auto",
            "--angle-count", "2",
            "--angle-step", "15",
            "--detector-count", "2",
            "--detector-spacing", "40",
            "--sample-count-y", "5",
            "--sample-count-z", "5",
            "--detector-resolution", "20",
            "--duration", "5",
            "--dt", "1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (
        "Actual STL pattern-family screening demo"
        in result.stdout
    )
    assert (
        "Actual STL pattern-family thermal-risk screening smoke check"
        in result.stdout
    )
    assert "Candidate count:" in result.stdout
    assert "Guardrail scenarios:" in result.stdout
    assert "Candidate:" in result.stdout
    assert "Body mask selected vertices:" in result.stdout
    assert "Pattern moved vertices:" in result.stdout
    assert "Scenario:" in result.stdout
    assert "Tradeoff label:" in result.stdout
    assert "Passes guardrails:" in result.stdout
    assert "Delta max temperature:" in result.stdout
    assert "Delta threshold count:" in result.stdout
    assert "Screening summary" in result.stdout
    assert "Strict passes:" in result.stdout
    assert "Measurement tolerance passes:" in result.stdout
    assert "Relaxed passes:" in result.stdout
    assert "Mixed candidates:" in result.stdout
    assert "Invariants: PASS" in result.stdout
