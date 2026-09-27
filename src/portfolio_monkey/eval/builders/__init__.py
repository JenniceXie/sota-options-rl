"""Panel builders: every path from a raw artifact to a :class:`NavPanel`.

Each builder is responsible for its own provenance. A builder may fail loudly
(coverage errors) but must never patch a gap, because a patched gap is an
undocumented modelling choice that would silently move a reported number.
"""

from portfolio_monkey.eval.builders import from_series

__all__ = ["from_series"]
