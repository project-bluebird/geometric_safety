"""Load a selected library before importing the current validation harness.

Workers run this file with ``-I`` so neither the working directory nor an
inherited PYTHONPATH can select a different library or an archived paper package.
"""

import argparse
import importlib
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
import runpy
import sys
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


def worker_command(module: str, arguments: list[str], source: Path | None = None) -> list[str]:
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--module", module]
    if source is not None:
        command.extend(("--source", str(source.resolve())))
    return [*command, "--", *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        choices=(
            "paper.benchmark_library_versions",
            "paper.validate_performance_parity",
            "paper.validate_directional_production",
        ),
        required=True,
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = (args.source or Path(__file__).resolve().parents[1]).resolve()
    expected = source / "geometric_safety" / "__init__.py"
    if not expected.is_file():
        raise ValueError(f"Selected library has no geometric_safety package: {source}")
    # Raising on a missing child prevents an editable-install finder from
    # silently completing this package from a different checkout.
    sys.meta_path.insert(0, _SelectedPackageFinder(source))
    sys.path.insert(0, str(source))
    library = importlib.import_module("geometric_safety")
    if library.__file__ is None or Path(library.__file__).resolve() != expected:
        raise RuntimeError("Worker imported a library outside the selected source")
    # All library imports now stay in the selected package. Prefer the current
    # harness for paper.* imports, including when an old checkout contains paper/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    arguments: list[str] = args.arguments
    sys.argv = [args.module, *(arguments[1:] if arguments[:1] == ["--"] else arguments)]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
