"""Strategy blocks.

Each module in this package defines one strategy and registers it. To add or replace a strategy,
add a module here (see base.py for the interface); tuning, estimation, the portfolio and the page
pick it up without any other change.
"""
import importlib
import pkgutil

from .base import ENGINE_OPTIONS, REGISTRY, SignalStrategy, Strategy, carry_diff, register

for _module in pkgutil.iter_modules(__path__):
    if _module.name != "base":
        importlib.import_module(f"{__name__}.{_module.name}")

# Every registered block, in display order.
STRATEGIES: dict[str, Strategy] = dict(sorted(REGISTRY.items(), key=lambda kv: (kv[1].order, kv[0])))

__all__ = ["ENGINE_OPTIONS", "STRATEGIES", "SignalStrategy", "Strategy", "carry_diff", "register"]
