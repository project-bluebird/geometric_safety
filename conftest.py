"""Keep compiled test code independent of persistent Numba caches."""

import os
from tempfile import TemporaryDirectory

import numba
import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Numba's cache does not invalidate callers after an imported callee changes.
    # A fresh cache tests this source tree, while allowing reuse within the run.
    cache = TemporaryDirectory(prefix="geometric-safety-pytest-numba-")
    previous_environment = os.environ.get("NUMBA_CACHE_DIR")

    os.environ["NUMBA_CACHE_DIR"] = cache.name
    numba.config.reload_config()

    def restore() -> None:
        if previous_environment is None:
            os.environ.pop("NUMBA_CACHE_DIR", None)
        else:
            os.environ["NUMBA_CACHE_DIR"] = previous_environment
        numba.config.reload_config()
        cache.cleanup()

    config.add_cleanup(restore)
