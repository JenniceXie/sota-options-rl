"""Interaction tables for the fixed-strategy baselines: family x horizon x name.

Self-contained on purpose. ``equal_weight_baselines.py`` is being edited by
another process and now filters to a single tenor bucket; this script needs to
composite *across* buckets to show the horizon dimension, so it duplicates the
few helpers rather than importing a moving target.

Every composite is **daily-rebalanced equal weight**: the arithmetic mean of the
constituents' simple returns, never the mean of their levels (which would let a
winner's growing weight carry the series) and never the mean of their log
returns (which is not a portfolio anyone can hold).

``WR``/``PLR`` are the pooled trade population of the constituent arms, read
from ``fills.realized`` -- net of cost and tied to NAV. A composite has no
trades of its own.

**A horizon here is a longer-dated CONTRACT, not a longer HOLD.** The holding
rule (``hold_steps=4``, ``min_dte=1``) is identical in every bucket, so the
tenor changes the contract's DTE -- and with it gamma, theta and vega -- while
the mean holding period stays ~3-4 days. The HP column makes that visible.

Usage::

    .venv/bin/python scripts/analysis/interaction_tables.py \\
        --runs "$PM_ARM_ROOT/baselines_pername"
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

TRADING_DAYS = 252
NAMES = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA", "SPY")
BASE = ("ol", "dv", "cv", "dg", "bf", "ls", "lg", "ib", "ic")
TENORS = ("0_7", "8_30", "31_90", "91_180")


def _d(t: str) -> date:
    return datetime.strptime(t.strip(), "%Y-%m-%d").date()


def metrics(levels: list[float]) -> dict:
    """TR, CAGR, AVOL, MDD, ASR, ACR, ASoR from a level path (simple returns)."""
    from math import sqrt
    from statistics import fmean, stdev

    n = len(levels) - 1
    if n < 2 or any(v <= 0 for v in levels):
        return {}
    r = [levels[i] / levels[i - 1] - 1.0 for i in range(1, len(levels))]
    years = n / TRADING_DAYS
    tr = levels[-1] / levels[0] - 1.0
    cagr = (levels[-1] / levels[0]) ** (1.0 / years) - 1.0
    peak = levels[0]
    mdd = 0.0
    for v in levels:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    sd = stdev(r)
    mu = fmean(r)
    down = sqrt(fmean([min(x, 0.0) ** 2 for x in r]))
    root = sqrt(TRADING_DAYS)
    return {
        "tr": tr,
        "cagr": cagr,
        "vol": sd * root if sd > 0 else None,
        "mdd": mdd,
        "asr": mu / sd * root if sd > 0 else None,
        "acr": cagr / mdd if mdd > 0 else None,
        "asor": mu / down * root if down > 0 else None,
    }


def load(runs: Path, grid: list[date]):
    """``{(name, cell, tenor): (returns, run_path)}`` for healthy arms.

    Keyed by the full cell (``olb``/``olr``), never by the two-character base:
    the two orientations are different arms and keying on the base silently
    overwrites one with the other -- 560 arms collapsing to 360 was the tell.
    """
    out, bad = {}, []
    for run in sorted(d for d in runs.iterdir() if d.is_dir()):
        panel = run / "nav_panel.jsonl"
        man = run / "manifest.json"
        if not panel.exists() or not man.exists():
            continue
        parts = run.name.split(".")
        if len(parts) < 5 or parts[1] != "fix":
            continue
        try:
            manifest = json.loads(man.read_text())
        except json.JSONDecodeError:
            bad.append((run.name, "unparseable manifest"))
            continue
        rows = [json.loads(l) for l in panel.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step_ts"])
        per_day = {_d(r["step_ts"][:10]): r["nlv"] for r in rows}
        days = sorted(per_day)
        if manifest.get("failure") or days[0] != grid[0] or days[-1] != grid[-1]:
            bad.append((run.name, manifest.get("failure") or f"span {days[0]}..{days[-1]}"))
            continue
        prev = manifest["start_nav"]
        rets = []
        for d in grid:
            cur = per_day.get(d, prev)
            rets.append(cur / prev - 1.0 if prev else 0.0)
            prev = cur
        out[(parts[0], parts[2], parts[3])] = (rets, run)
    return out, bad


def compose(series: list[list[float]]) -> list[float]:
    levels = [1.0]
    for i in range(len(series[0])):
        levels.append(levels[-1] * (1.0 + sum(s[i] for s in series) / len(series)))
    return levels


def trades(paths: list[Path]) -> tuple[float | None, float | None]:
    wins, losses = [], []
    for run in paths:
        fp = run / "fills.jsonl"
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            f = json.loads(line)
            if f.get("kind") not in ("close", "expire"):
                continue
            r = f.get("realized")
            if r is None:
                continue
            (wins if r > 0 else losses if r < 0 else []).append(abs(r))
    decided = len(wins) + len(losses)
    wr = len(wins) / decided if decided else None
    plr = (
        (sum(wins) / len(wins)) / (sum(losses) / len(losses)) if wins and losses else None
    )
    return wr, plr


HEAD = (
    f"{'cut':<10}{'TR%':>9}{'ASR':>8}{'ACR':>8}{'ASoR':>8}{'AVOL%':>9}"
    f"{'MDD%':>8}{'WR%':>8}{'PLR':>8}{'n':>6}"
)


def emit(label: str, keys, arms) -> None:
    got = [arms[k][0] for k in keys if k in arms]
    if not got:
        return
    m = metrics(compose(got))
    wr, plr = trades([arms[k][1] for k in keys if k in arms])

    def f(v, s=1.0, p=2):
        return "--" if v is None else f"{v * s:.{p}f}"

    print(
        f"{label:<10}{100 * m['tr']:>9.2f}{f(m['asr']):>8}{f(m['acr']):>8}{f(m['asor']):>8}"
        f"{f(m['vol'], 100):>9}{100 * m['mdd']:>8.2f}{f(wr, 100):>8}{f(plr):>8}{len(got):>6}"
    )


def matrix(title: str, rows, cols, keyfn, arms) -> None:
    print(f"\n{title}")
    head = f"{'':<8}" + "".join(f"{c:>9}" for c in cols) + f"{'ALL':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r:<8}"
        for c in list(cols) + [None]:
            keys = [k for k in arms if keyfn(k, r, c)]
            if not keys:
                line += f"{'--':>9}"
                continue
            m = metrics(compose([arms[k][0] for k in keys]))
            line += f"{100 * m['tr']:>9.1f}"
        print(line)
    line = f"{'ALL':<8}"
    for c in list(cols) + [None]:
        keys = [k for k in arms if any(keyfn(k, r, c) for r in rows)]
        m = metrics(compose([arms[k][0] for k in keys])) if keys else {}
        line += f"{100 * m['tr']:>9.1f}" if m else f"{'--':>9}"
    print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--dates", type=Path, default=Path("configs/dates_eval.txt"))
    p.add_argument(
        "--tenor-only",
        default="",
        help="restrict every cut to one horizon bucket, for the cross-tabulation "
        "actually reported; omit to composite across all buckets",
    )
    args = p.parse_args()

    grid = [_d(l) for l in args.dates.read_text().splitlines() if l.strip()]
    arms, bad = load(args.runs, grid)
    if args.tenor_only:
        arms = {k: v for k, v in arms.items() if k[2] == args.tenor_only}
    print(f"window   {grid[0]} .. {grid[-1]}  ({len(grid)} dates)")
    print(f"arms     {len(arms)} healthy, {len(bad)} rejected")
    for b in bad[:5]:
        print(f"   REJECT {b}")
    print("method   daily-rebalanced equal weight; WR/PLR pooled from fills.realized")
    print("horizon  = contract DTE, NOT holding period (hold rule is the same in all buckets)")

    tenors = [t for t in TENORS if any(k[2] == t for k in arms)]

    print("\n== MARGINAL: by STRATEGY FAMILY (all names, orientations, horizons) ==")
    print(HEAD)
    print("-" * len(HEAD))
    for b in BASE:
        emit(b, [k for k in arms if k[1][:2] == b], arms)

    print("\n== MARGINAL: by HORIZON (all names, families) ==")
    print(HEAD)
    print("-" * len(HEAD))
    for t in tenors:
        emit(t, [k for k in arms if k[2] == t], arms)

    print("\n== MARGINAL: by NAME (all families, horizons) ==")
    print(HEAD)
    print("-" * len(HEAD))
    for n in NAMES:
        emit(n, [k for k in arms if k[0] == n], arms)

    print("\n== GRAND ==")
    print(HEAD)
    print("-" * len(HEAD))
    emit("ALL", list(arms), arms)

    matrix(
        "INTERACTION  total return %  --  FAMILY (rows) x HORIZON (cols)",
        BASE, tenors,
        lambda k, r, c: k[1][:2] == r and (c is None or k[2] == c),
        arms,
    )
    matrix(
        "INTERACTION  total return %  --  FAMILY (rows) x NAME (cols)",
        BASE, NAMES,
        lambda k, r, c: k[1][:2] == r and (c is None or k[0] == c),
        arms,
    )
    matrix(
        "INTERACTION  total return %  --  NAME (rows) x HORIZON (cols)",
        NAMES, tenors,
        lambda k, r, c: k[0] == r and (c is None or k[2] == c),
        arms,
    )


if __name__ == "__main__":
    main()
