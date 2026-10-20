"""Load a selected library before importing the current validation harness.

Workers run this file with ``-I`` so neither the working directory nor an
inherited PYTHONPATH can select a different library or an archived paper package.
"""

import argparse
import importlib
from pathlib import Path
import runpy
import sys


def worker_command(module: str, arguments: list[str], source: Path | None = None) -> list[str]:
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--module", module]
    if source is not None:
        command.extend(("--source", str(source.resolve())))
    return [*command, "--", *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module", choices=("paper.benchmark_library_versions", "paper.validate_performance_parity"), required=True
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    expected: Path | None = None
    if args.source is not None:
        source: Path = args.source
        expected = source.resolve() / "geometric_safety" / "__init__.py"
        if not expected.is_file():
            raise ValueError(f"Selected library has no geometric_safety package: {args.source}")
        sys.path.insert(0, str(source.resolve()))
    library = importlib.import_module("geometric_safety")
    if expected is not None and (library.__file__ is None or Path(library.__file__).resolve() != expected):
        raise RuntimeError("Worker imported a library outside the selected source")
    # The selected package and its __path__ are now fixed. Prefer the current
    # harness for paper.* imports, including when an old checkout contains paper/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    arguments: list[str] = args.arguments
    sys.argv = [args.module, *(arguments[1:] if arguments[:1] == ["--"] else arguments)]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
