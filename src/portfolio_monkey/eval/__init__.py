"""Evaluation pipeline for ``docs/evaluation_protocol.md``.

The package is deliberately split so that the path from raw artifacts to a NAV
panel (``builders``) cannot influence the reduction from a panel to a number
(``metrics``). State variables change across experiments; reported metrics must
not move with them.
"""

from portfolio_monkey.eval import arms, metrics, schema

__all__ = ["arms", "metrics", "schema"]
