"""Per-package binary targets: will *this* structure make money on this name today?

This replaces the argmax framing and the reason is measured, not stylistic.
Asking "which of the 13 heads is best for this cell" makes a label that is
`outright` 96.9% of the time, because with hindsight an uncapped directional
position beats a defined-risk one whenever the direction is known.  A model
fitted to that learns the constant: on the RL window it predicts `ol` for 568 of
570 rows, recall 1.00 on the majority class and 0.00 on every other, and a
permutation control scores *better* than it does.

Asking instead, for each head separately, "is this package's realized
after-cost PnL positive" gives one binary problem per head, with its own base
rate, and those base rates are not degenerate: `ib n` 43.6%, `dv r` 47.0%,
`ic n` 47.2% are near coin-flip, while `bf r` 9.7% says butterflies almost never
pay.  None of that is visible through an argmax.

**The unit is a parametrized package, not a family.**  One row is
``(trade_date, underlying, head, tenor)``.  ``head`` carries the orientation
because for `ol` the orientation *is* the bet, and the coordinates come from the
resolver's family defaults exactly as they do for every rule arm -- so the label
is about a package the environment would actually build, at the size the
``SizeResolver`` would actually choose.

**Tenor is a feature, not a separate model.**  The four tenors of one cell share
one market feature vector, so they are four different packages over the same
state.  That is also the honest sample-size caveat: n is ~2,300 rows per head
but the independent units are the ~610 cells, so an interval computed from the
row count is about four times too tight.

**What was found, train SFT / test RL, one model per head with a permutation
control each:** mean AUC 0.561 for gradient boosting against 0.497 for the
shuffled control, and the split is economic rather than uniform ---

    volatility  bf r .642  lg n .628  ib n .623  ic n .616  bf b .595  ls n .589
    directional ol b .553  dg b .542  ol r .526  dg r .522  dv b .521  cv b .504
                cv r .499  dv r .495

Implied level, skew, curvature and term slope predict whether a *volatility*
structure pays, because that is a bet on realized against implied and those
fields are what measure it.  They do not predict direction.  The predictable set
is exactly the five families ``--hedge volatility`` hedges, which is a
consistency check falling out of an unrelated calculation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import Frame, LeakageError

#: Tenor bucket -> ordinal. Ordered because the buckets are ordered in days, so
#: a tree can split on "longer than a month" with one threshold instead of
#: needing three one-hot columns to express it.
TENOR_INDEX: Mapping[str, int] = {"0_7": 0, "8_30": 1, "31_90": 2, "91_180": 3}

#: The five families the hedge resolver covers, i.e. the volatility structures.
#: Named here so the volatility/directional split in the results can be
#: reproduced without retyping it, and imported from the env rather than
#: restated so it cannot drift from the hedge spec.
def volatility_heads(heads: Sequence[str]) -> tuple[str, ...]:
    """The heads whose family is one of the five volatility structures.

    Reads ``VOLATILITY_FAMILIES`` and **not** ``HedgeSpec().hedged_families``:
    that field defaults to the empty tuple and is only populated at runtime by
    ``--hedge volatility``, so a helper built on it returns nothing and the
    volatility/directional split silently becomes a split of an empty set
    against everything.  That is what the first version of this did.
    """
    from ..env.actions import FAMILY_CODES
    from ..env.spec import VOLATILITY_FAMILIES

    codes = {name: code for code, name in FAMILY_CODES.items()}
    wanted = {codes[name] for name in VOLATILITY_FAMILIES if name in codes}
    if not wanted:
        raise LeakageError(
            "VOLATILITY_FAMILIES did not map onto any wire code; the family "
            "tables have diverged and the split below would be meaningless"
        )
    return tuple(h for h in heads if h.split()[0] in wanted)


@dataclass(frozen=True, slots=True)
class PackageSet:
    """One binary problem per head, aligned features and labels."""

    heads: tuple[str, ...]
    X: Mapping[str, np.ndarray]
    y: Mapping[str, np.ndarray]
    dates: Mapping[str, np.ndarray]
    feature_names: tuple[str, ...]

    def base_rate(self, head: str) -> float:
        return float(self.y[head].mean())

    def summary(self) -> list[dict]:
        return [
            {
                "head": h,
                "n": int(len(self.y[h])),
                "positive_rate": self.base_rate(h),
                "dates": int(len(set(self.dates[h].tolist()))),
            }
            for h in self.heads
        ]


def build(
    dump: Path,
    features: Frame,
    *,
    min_rows: int = 200,
) -> PackageSet:
    """Join a candidate dump to a feature frame, one binary problem per head.

    ``features`` supplies the market state per ``(trade_date, underlying)``; the
    dump supplies one candidate per ``(date, underlying, head, tenor)`` with its
    realized after-cost PnL.  A dump row whose cell is absent from ``features``
    is dropped rather than imputed -- it means the two artifacts cover different
    windows, which is a mistake to surface and not to paper over.

    ``min_rows`` drops heads too thin to fit or to score. It is a floor on rows,
    not on cells, and the docstring's sample-size caveat still applies above it.
    """
    cells = {
        (r.trade_date, r.underlying): [r.features[n] for n in features.feature_names]
        for r in features.rows
    }
    if not cells:
        raise LeakageError(
            "the feature frame is empty; a package set built from it would be "
            "silently empty too"
        )
    acc: dict[str, list[tuple[list[float], int, str]]] = {}
    unmatched = 0
    with dump.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["trade_date"], row["underlying"])
            base = cells.get(key)
            if base is None:
                unmatched += 1
                continue
            tenor = TENOR_INDEX.get(row["tenor"])
            if tenor is None:
                continue
            acc.setdefault(row["head"], []).append(
                (
                    [np.nan if v is None else float(v) for v in base] + [float(tenor)],
                    1 if row["profit"] > 0 else 0,
                    row["trade_date"],
                )
            )
    heads = tuple(
        h for h in sorted(acc)
        if len(acc[h]) >= min_rows and len({lab for _, lab, _ in acc[h]}) == 2
    )
    return PackageSet(
        heads=heads,
        X={h: np.array([r[0] for r in acc[h]], dtype=float) for h in heads},
        y={h: np.array([r[1] for r in acc[h]], dtype=int) for h in heads},
        dates={h: np.array([r[2] for r in acc[h]], dtype=object) for h in heads},
        feature_names=tuple(features.feature_names) + ("tenor_index",),
    )


def _model(backend: str, seed: int = 0):
    if backend == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )
    if backend == "gbdt":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=4, tree_method="hist",
        )
    if backend == "mlp":
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Two small hidden layers and early stopping, sized against the data
        # rather than against convention.  The independent units here are the
        # ~610 (date, underlying) cells, not the ~2,300 rows -- the four tenors
        # of a package share one feature vector -- so (32, 16) already has more
        # parameters than cells.  Anything wider is fitting the window, and the
        # permutation control is what will say so.
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-2,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.15,
                random_state=seed,
            ),
        )
    raise ValueError(f"unknown backend {backend!r}")


def evaluate(
    train: PackageSet,
    test: PackageSet,
    *,
    backend: str = "gbdt",
    seed: int = 0,
    with_control: bool = True,
) -> list[dict]:
    """Per-head AUC on ``test``, each with its own permutation control.

    The control is not decoration.  A binary problem whose base rate differs
    between windows can produce an AUC above 0.5 from the prior alone, and the
    argmax formulation this replaces was defeated by exactly that: its shuffled
    control scored *higher* than the real model.  Fitting the same model on
    labels shuffled within the training window gives the number that a model
    with no signal would earn on this particular pair of windows, and it came
    back at 0.497 -- so the gap is interpretable.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for head in train.heads:
        if head not in test.heads:
            continue
        Xt, yt = train.X[head], train.y[head]
        Xv, yv = test.X[head], test.y[head]
        auc = roc_auc_score(yv, _model(backend, seed).fit(Xt, yt).predict_proba(Xv)[:, 1])
        row = {
            "head": head,
            "n_train": int(len(yt)),
            "n_test": int(len(yv)),
            "base_rate_train": float(yt.mean()),
            "base_rate_test": float(yv.mean()),
            "auc": float(auc),
        }
        if with_control:
            shuffled = yt[rng.permutation(len(yt))]
            row["auc_shuffled"] = float(
                roc_auc_score(
                    yv, _model(backend, seed).fit(Xt, shuffled).predict_proba(Xv)[:, 1]
                )
            )
            row["beats_control"] = row["auc"] > row["auc_shuffled"]
        out.append(row)
    return out
