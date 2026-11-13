"""The v9 supervisor graph: a team of specialists with a manager. See graph.py.

WHY THE IMPORT BELOW IS LAZY
    graph.py imports every specialist, and the orders specialist imports
    supervisor/handoffs.py. Importing ANY module of this package runs this file
    first, so an eager `from workflows.supervisor.graph import ...` here made a
    loop: orders.py -> handoffs.py -> this file -> graph.py -> orders.py, which
    is only half loaded, and `import workflows.specialists.orders` failed unless
    something had imported the graph before it. A module-level __getattr__
    (PEP 562) loads graph.py only when build_team or open_team is asked for, so
    `from workflows.supervisor import open_team` still works, in any order.
"""
from __future__ import annotations

__all__ = ["build_team", "open_team"]


def __getattr__(name: str):
    if name in __all__:
        from workflows.supervisor import graph
        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
