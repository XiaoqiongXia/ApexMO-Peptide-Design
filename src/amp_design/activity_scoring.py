"""Compatibility alias for :mod:`amp_design.predictors.activity`.

New code should import the functional module directly.
"""
import importlib
import sys

_implementation = importlib.import_module("amp_design.predictors.activity")
if __name__ == "__main__":
    if hasattr(_implementation, "main"):
        _implementation.main()
else:
    sys.modules[__name__] = _implementation
