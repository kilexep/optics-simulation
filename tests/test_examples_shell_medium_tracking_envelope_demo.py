import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_shell_medium_tracking_envelope_demo.py"
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
    assert "Shell medium tracking envelope demo" in result.stdout
    assert (
        "Synthetic shell medium-tracking validity smoke check"
        in result.stdout
    )
    assert "Paraxial envelope" in result.stdout
    assert "Wide envelope" in result.stdout
    assert "epsilon" in result.stdout
    assert "Fill medium: air" in result.stdout
    assert "Fill medium: water" in result.stdout
    assert "Validation passed:" in result.stdout
    assert "Unexpected surface count:" in result.stdout
    assert "Envelope diagnostic" in result.stdout
    assert "Invariants: PASS" in result.stdout
