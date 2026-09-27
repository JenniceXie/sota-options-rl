"""Fit the per-package classifiers and freeze them, thresholds included.

Nothing about this artifact is decided at run time.  The classifiers are fitted
on the SFT window, the abstain threshold is chosen on the RL window, and both are
written to disk with a provenance block, so "was this tuned on the test set?" is
answered by reading one file rather than by trusting a policy's docstring.

The threshold is selected on **validation precision at a fixed trade rate**, not
on accuracy or AUC.  AUC is what the model was selected on and is invariant to
the threshold, so it cannot choose one; accuracy would pick whichever threshold
abstains most, because most heads have a base rate far from 0.5.  What the
trading arm needs is the level above which the ranked candidates are more often
right than the head's own base rate, which is what this reports.

Usage:
    python scripts/analysis/fit_package_models.py OUT_DIR \
        --train-features <hold run over dates_sft> \
        --train-dump <oracle candidates for dates_sft> \
        --val-features <hold run over dates_rl> \
        --val-dump <oracle candidates for dates_rl> \
        --repo .
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np

from portfolio_monkey.baselines import dataset as D
from portfolio_monkey.baselines import packages as P

#: Candidate abstain thresholds, in units of **lift over the head's own base
#: rate**, not raw probability.  1.0 means "the model is no more confident than
#: the unconditional rate", so everything here asks for some edge.
#:
#: Raw-probability thresholds were tried first and are a trap: a single global
#: cut across heads whose base rates run 0.10 to 0.84 selects the high-base-rate
#: heads and nothing else.  At 0.70 it picked exactly the four with the highest
#: base rates, all with AUC 0.50-0.55, and excluded every head with real signal.
#: Coarse on purpose: a finer grid chosen on 57 validation dates of one market
#: path would be fitting the grid.
THRESHOLDS: tuple[float, ...] = (1.00, 1.05, 1.10, 1.20, 1.30)

#: Mirrors ``policy.ABSOLUTE_FLOOR``; the scan must score what the arm will do.
ABSOLUTE_FLOOR = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--train-features", required=True, type=Path)
    p.add_argument("--train-dump", required=True, type=Path)
    p.add_argument("--val-features", required=True, type=Path)
    p.add_argument("--val-dump", required=True, type=Path)
    p.add_argument("--backend", default="gbdt", choices=("gbdt", "logistic"))
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train = P.build(
        args.train_dump,
        D.load_feature_frame(args.train_features, repo=args.repo, split="train"),
        min_rows=args.min_rows,
    )
    val = P.build(
        args.val_dump,
        D.load_feature_frame(args.val_features, repo=args.repo, split="val"),
        min_rows=args.min_rows,
    )
    heads = tuple(h for h in train.heads if h in val.heads)
    if not heads:
        raise SystemExit("no head survives in both windows; nothing to fit")

    print(f"heads {len(heads)}  features {len(train.feature_names)}  backend {args.backend}")
    models = {}
    for head in heads:
        models[head] = P._model(args.backend, args.seed).fit(
            train.X[head], train.y[head]
        )

    # -- threshold, on validation only -----------------------------------
    #
    # Pooled across heads, because the arm ranks candidates against each other
    # and so needs one common cut, not fourteen.
    base_rates = {h: float(train.y[h].mean()) for h in heads}
    probs, labels, lifts, which = [], [], [], []
    for head in heads:
        p = models[head].predict_proba(val.X[head])[:, 1]
        probs.append(p)
        labels.append(val.y[head])
        lifts.append(p / base_rates[head])
        which.append(np.full(len(p), head, dtype=object))
    prob = np.concatenate(probs)
    lab = np.concatenate(labels)
    lift = np.concatenate(lifts)
    head_of = np.concatenate(which)
    base = float(lab.mean())
    floor_ok = prob > ABSOLUTE_FLOOR
    rows = []
    for t in THRESHOLDS:
        keep = floor_ok & (lift > t)
        picked = sorted(set(head_of[keep].tolist())) if keep.any() else []
        rows.append(
            {
                "threshold": t,
                "kept_fraction": float(keep.mean()),
                "precision": float(lab[keep].mean()) if keep.any() else None,
                "lift_over_base": (
                    float(lab[keep].mean() - base) if keep.any() else None
                ),
                # Which heads survive is the diagnostic that would have caught
                # the base-rate sort immediately: under the old rule this list
                # was the four highest-base-rate heads at every threshold.
                "heads_selected": picked,
                "n_heads_selected": len(picked),
            }
        )
        k = rows[-1]
        prec = "   -  " if k["precision"] is None else f"{k['precision']:.3f}"
        lft = "  -  " if k["lift_over_base"] is None else f"{k['lift_over_base']:+.3f}"
        print(
            f"  lift>{t:.2f}  keeps {k['kept_fraction']:6.1%}  precision {prec}"
            f"  vs base {lft}  heads {k['n_heads_selected']:2d}"
        )
    usable = [r for r in rows if r["precision"] is not None and r["kept_fraction"] >= 0.01]
    if not usable:
        raise SystemExit(
            "every candidate threshold abstains on more than 99% of validation "
            "packages; refusing to freeze a threshold that would make the arm a "
            "hold arm wearing a model's name"
        )
    best = max(usable, key=lambda r: r["lift_over_base"])
    print(f"selected threshold {best['threshold']:.2f}  "
          f"lift {best['lift_over_base']:+.3f} over base {base:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    for head, model in models.items():
        (args.out / f"{head.replace(' ', '_')}.pkl").write_bytes(pickle.dumps(model))
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "artifact": "package_models",
                "built": date.today().isoformat(),
                "backend": args.backend,
                "seed": args.seed,
                "heads": list(heads),
                "base_rates": base_rates,
                "absolute_floor": ABSOLUTE_FLOOR,
                "threshold_units": (
                    "lift over the head's own training base rate, not raw "
                    "probability. A global probability cut is a base-rate sort: "
                    "at 0.70 it selected exactly the four highest-base-rate "
                    "heads, every one of them with AUC 0.50-0.55, and excluded "
                    "all five heads with real signal."
                ),
                "feature_names": list(train.feature_names),
                "tenors": sorted(P.TENOR_INDEX, key=P.TENOR_INDEX.get),
                "threshold": best["threshold"],
                "threshold_scan": rows,
                "validation_base_rate": base,
                "fitted_on": {
                    "train_features": str(args.train_features),
                    "train_dump": str(args.train_dump),
                    "rows_per_head": {h: int(len(train.y[h])) for h in heads},
                },
                "threshold_chosen_on": {
                    "val_features": str(args.val_features),
                    "val_dump": str(args.val_dump),
                },
                "test_window": (
                    "NOT TOUCHED. Neither the classifiers nor the threshold has "
                    "seen configs/dates_eval.txt."
                ),
                "label_caveat": (
                    "The target is the sign of the oracle's best-exit PnL, so "
                    "P(positive) is 'could this package have paid if exited "
                    "well'. A fixed exit rule cannot realize that, so trading "
                    "PnL will sit below what the AUC suggests and the gap "
                    "belongs to the exit rule."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
