"""Python-only call instrumentation for benchmarks and numerical studies.

Copies use private global namespaces; production dispatchers are never replaced.
The solver does not import this module.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from types import FunctionType


def python_function(dispatcher: object) -> FunctionType:
    """Recover a Python function from a callable or Numba dispatcher."""
    if not callable(dispatcher):
        raise TypeError("Numerical instrumentation requires a callable")
    function = inspect.unwrap(dispatcher)
    if not isinstance(function, FunctionType):
        raise TypeError("Numerical instrumentation requires a Python function")
    return function


def copy_python_function(dispatcher: object, overrides: Mapping[str, object]) -> FunctionType:
    """Copy a function with selected globals replaced in its own namespace."""
    original = python_function(dispatcher)
    namespace = original.__globals__ | dict(overrides)
    copied = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    copied.__kwdefaults__ = original.__kwdefaults__
    return copied


@dataclass
class CallCounter:
    """Count calls to one real dependency while preserving its return value."""

    target: Callable[..., object]
    calls: int = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self.target(*args, **kwargs)


def counted_python_function(dispatcher: object, dependency: str) -> tuple[FunctionType, CallCounter]:
    """Count direct calls to a named global dependency in a private Python copy.

    Other dependencies remain unchanged, including compiled helpers. This measures
    Python control flow, which can differ from compiled flow at rounding boundaries.
    """
    original = python_function(dispatcher)
    target = original.__globals__[dependency]
    if not callable(target):
        raise TypeError(f"Cannot count non-callable dependency {dependency}")
    counter = CallCounter(target)
    return copy_python_function(original, {dependency: counter}), counter
