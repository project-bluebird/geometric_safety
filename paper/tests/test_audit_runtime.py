"""Audits must execute the selected source even after an imported helper changes."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from geometric_safety import relevant_aircraft as ra


@pytest.mark.parametrize("preserve_bytecode", [False, True])
def test_numerical_audit_observes_a_helper_edit_in_the_same_interpreter(
    tmp_path: Path, preserve_bytecode: bool
) -> None:
    package = tmp_path / "geometric_safety"
    package.mkdir()
    for source in Path(ra.__file__).parent.glob("*.py"):
        shutil.copy2(source, package / source.name)
    code = """
import os
from pathlib import Path
import py_compile
import subprocess
import sys
sys.path.insert(0, sys.argv[1])
import geometric_safety
sys.path.insert(0, sys.argv[2])
from paper.validate_directional_production import numerical_checks
numerical_checks(11, 2026092031)
utility = Path(geometric_safety.__file__).with_name('util.py')
source = utility.read_bytes()
original = b'heading_rad = heading_deg * DEG_TO_RAD'
assert source.count(original) == 1
changed = b'heading_rad=(heading_deg+5)*DEG_TO_RAD'
assert len(changed) == len(original)
metadata = utility.stat()
if sys.argv[3] == 'True':
    py_compile.compile(str(utility), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
utility.write_bytes(source.replace(original, changed))
if sys.argv[3] == 'True':
    os.utime(utility, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert utility.stat().st_mtime_ns == metadata.st_mtime_ns
    assert utility.stat().st_size == metadata.st_size
try:
    numerical_checks(11, 2026092031)
except (RuntimeError, subprocess.CalledProcessError):
    print('changed helper rejected by the numerical audit')
else:
    raise RuntimeError('audit reused the original imported helper')
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(tmp_path),
            str(Path(__file__).resolve().parents[2]),
            str(preserve_bytecode),
        ],
        env=dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "inherited-cache")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "changed helper rejected by the numerical audit" in completed.stdout
    assert "Directional validation" in completed.stderr
