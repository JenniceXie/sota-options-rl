"""Conventional machine-learning baselines: market state -> strategy family.

Band D of the paper's matrix.  The contract these share with every other arm is
that the *only* thing learned is the family choice: once a family is picked, the
same deterministic resolver every rule-based arm uses turns it into strikes, a
tenor and a size.  Nothing here predicts a DTE, a delta coordinate or a package
count, because a model that did would be a different environment.

Two label sources live here and they are **never mixed**, because they answer
different questions:

``teacher``
    What family did the GPT teacher open, given this state?  An *imitation*
    target.  The ceiling is the teacher, and the teacher loses money.

``oracle``
    Which family would have made the most after-cost money, given this state?
    A *prediction* target, and the one the paper's Band D spec declares
    (``target: realized_after_cost_pnl``).

Both are reported; the gap between them is the finding.
"""

from __future__ import annotations

__all__ = ["dataset"]
