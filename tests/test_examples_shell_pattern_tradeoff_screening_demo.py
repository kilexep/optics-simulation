import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "run_shell_pattern_tradeoff_screening_demo.py"
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
        timeout=600,
        env=env,
    )

    assert result.returncode == 0, (
        f"demo exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Shell pattern tradeoff screening demo" in result.stdout
    assert (
        "Synthetic thermal-risk comparison and guardrail smoke check"
        in result.stdout
    )
    assert "Candidate count:" in result.stdout
    assert "Fill medium: air" in result.stdout
    assert "Fill medium: water" in result.stdout
    assert "Candidate:" in result.stdout
    assert "Tradeoff label:" in result.stdout
    assert "Passes guardrails:" in result.stdout
    assert "Guardrail violations:" in result.stdout
    assert "Delta max temperature:" in result.stdout
    assert "Delta threshold count:" in result.stdout
    assert "Screening summary" in result.stdout
    assert "Passed candidates:" in result.stdout
    assert "Failed candidates:" in result.stdout
    assert "Mixed candidates:" in result.stdout
    assert "Medium tracking validation passed:" in result.stdout
    assert "Unexpected surface count:" in result.stdout
    assert "Trace epsilon:" in result.stdout
    assert "Validated side y range:" in result.stdout
    assert "Validated side z range:" in result.stdout
    assert "Invariants: PASS" in result.stdout
