import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_actual_stl_ring_offset_sweep_demo.py"
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
            "--subdivision-iterations", "1",
            "--max-conditions", "2",
            "--angle-count", "2",
            "--angle-step", "15",
            "--detector-count", "2",
            "--detector-spacing", "40",
            "--sample-count-y", "5",
            "--sample-count-z", "5",
            "--detector-resolution", "20",
            "--pattern-count", "5",
            "--pattern-max-depth", "0.02",
            "--duration", "5",
            "--dt", "1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Actual STL ring-offset sweep demo" in result.stdout
    assert (
        "Actual STL ring-offset risk-guided pattern parameter "
        "sweep diagnostic"
        in result.stdout
    )
    assert "Candidate count:" in result.stdout
    assert "Selection condition count:" in result.stdout
    assert "Evaluation condition count:" in result.stdout
    assert "Holdout condition count:" in result.stdout
    assert "Source risk active count:" in result.stdout
    assert "Candidate:" in result.stdout
    assert "Inner radius:" in result.stdout
    assert "expansion ratio" in result.stdout
    assert "Moved vertices:" in result.stdout
    assert (
        "Full worst delta max temperature:" in result.stdout
    )
    assert (
        "In-sample worst delta max temperature:" in result.stdout
    )
    assert (
        "Holdout worst delta max temperature:" in result.stdout
    )
    assert "Hotspot overlap" in result.stdout
    assert "Source overlap bins:" in result.stdout
    assert "Candidate-only bins:" in result.stdout
    assert "new bin fraction" in result.stdout
    assert "Jaccard overlap:" in result.stdout
    assert (
        "Pareto non-dominated diagnostic set" in result.stdout
    )
    assert "Non-discriminative metrics" in result.stdout
    assert (
        "no_improving_candidate_under_current_sweep:"
        in result.stdout
    )
    assert (
        "no_improving_candidate_in_holdout:" in result.stdout
    )
    assert "holdout worst delta max temperature" in result.stdout
    assert "holdout threshold-count delta" in result.stdout
    assert "holdout pass ratio" in result.stdout
    assert "In-sample top by max-T delta:" in result.stdout
    assert "Holdout top by max-T delta:" in result.stdout
    assert (
        "Pass-ratio top vs worst-delta-T top" in result.stdout
        or "non-discriminative; comparison skipped"
        in result.stdout
    )
    assert "Result invariant validation" in result.stdout
    assert (
        "this is not manufacturing-ready validation"
        in result.stdout
    )
    assert (
        "in-sample improvement does not imply holdout "
        "improvement" in result.stdout
    )
    assert "Invariants: PASS" in result.stdout
