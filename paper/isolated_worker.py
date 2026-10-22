"""Load a selected library before importing the current validation harness.

Workers run this file with ``-I`` so neither the working directory nor an
inherited PYTHONPATH can select a different library or an archived paper package.
"""

import argparse
import importlib
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from importlib.util import find_spec
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType


class _SelectedPackageFinder(MetaPathFinder):
    """Resolve every library submodule exclusively inside the selected package."""

    def __init__(self, source: Path) -> None:
        self.source = source.resolve()
        self.package = self.source / "geometric_safety"

    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None) -> ModuleSpec | None:
        del path, target
        parts = fullname.split(".")
        if parts[0] != "geometric_safety":
            return None
        directory = self.source if len(parts) == 1 else self.package.joinpath(*parts[1:-1])
        spec = PathFinder.find_spec(fullname, [str(directory)])
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(self.package):
            raise ModuleNotFoundError(f"{fullname} is absent from the selected library: {self.source}", name=fullname)
        return spec


def worker_command(
    module: str, arguments: list[str], source: Path | None = None, *, inputs: tuple[Path, ...] = ()
) -> list[str]:
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--module", module]
    if source is not None:
        command.extend(("--source", str(source.resolve())))
    for path in inputs:
        command.extend(("--verify-input", str(path.resolve())))
    return [*command, "--", *arguments]


def run_isolated(module: str, arguments: list[str], source: Path) -> None:
    """Run one audit with strict package selection and fresh Python and compiler caches."""
    subprocess.run(worker_command(module, arguments, source), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        choices=(
            "paper.benchmark_library_versions",
            "paper.validate_performance_parity",
            "paper.validate_directional_production",
            "paper.run_turn_numerics",
            "paper.validate_performance_numerics",
        ),
        required=True,
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--verify-input", type=Path, action="append", default=[])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.source is None:
        # Resolve the interpreter's installed candidate before exposing the harness.
        spec = find_spec("geometric_safety")
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError("No installed geometric_safety package", name="geometric_safety")
        source = Path(spec.origin).resolve().parent.parent
    else:
        source = args.source.resolve()
    expected = source / "geometric_safety" / "__init__.py"
    if not expected.is_file():
        raise ValueError(f"Selected library has no geometric_safety package: {source}")
    # Raising on a missing child prevents an editable-install finder from
    # silently completing this package from a different checkout.
    sys.meta_path.insert(0, _SelectedPackageFinder(source))
    sys.path.insert(0, str(source))
    # All library imports now stay in the selected package. Prefer the current
    # harness for paper.* imports, including when an old checkout contains paper/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with TemporaryDirectory(prefix="geometric-safety-worker-") as cache:
        # -B prevents writes, but existing timestamp-valid .pyc files can still
        # hide a source edit. Both Python and Numba must start with empty caches.
        sys.pycache_prefix = str(Path(cache) / "python")
        os.environ["NUMBA_CACHE_DIR"] = str(Path(cache) / "numba")
        from paper.benchmark_contracts import digest, module_hashes

        package_hashes = module_hashes(expected.parent)
        harness_hashes = module_hashes(Path(__file__).parent)
        input_hashes = {path: digest(path) for path in args.verify_input}
        library = importlib.import_module("geometric_safety")
        if library.__file__ is None or Path(library.__file__).resolve() != expected:
            raise RuntimeError("Worker imported a library outside the selected source")
        arguments: list[str] = args.arguments
        sys.argv = [args.module, *(arguments[1:] if arguments[:1] == ["--"] else arguments)]
        runpy.run_module(args.module, run_name="__main__")
        if package_hashes != module_hashes(expected.parent):
            raise RuntimeError("Selected package changed during measurement")
        if harness_hashes != module_hashes(Path(__file__).parent):
            raise RuntimeError("Validation harness changed during measurement")
        if input_hashes != {path: digest(path) for path in args.verify_input}:
            raise RuntimeError("Selected inputs changed during measurement")


if __name__ == "__main__":
    main()
