"""Rebuild the main results table from run directories.

Every row is on the test window, and every baseline ran in the environment the
language policy was trained and evaluated in: ``paper_base_config()`` with vendor
mark quotes. The script does not trust that; it checks it (see *Validation*).

Rows
----
====================  ========================================================
SOTA                  one run directory (state-only arm, RL step 20)
Equal-weighted rule   the fixed-strategy arms at one contract horizon (default
                      8-30 DTE: 10 names x 14 family/orientation cells = 140
                      arms), composited by ``interaction_tables.py``
GARCH rule            one run directory (``--policy garch_vrp``)
Threshold rule        one run directory (``--policy econ_rules``)
GBDT, Logistic        one run directory each (``--policy package_model``)
Ex-post oracle        optional; the ``nav_curve`` written by
                      ``clairvoyant_oracle.py``. A hindsight bound, not a policy.
====================  ========================================================

Metrics
-------
TR, ASR, ACR, ASoR, AVOL, MDD
    ``eval.builders.from_ledger.build_ledger_panel`` on the PM grid, then
    ``eval.metrics.compute`` (log returns, risk-free rate 0). The oracle's
    ``nav_curve`` is rebuilt into the same ``NavPanel`` and goes through the same
    ``compute``; never use the ``performance`` block inside the oracle JSON, which
    is computed from simple returns. The equal-weighted row is a daily-rebalanced
    composite, not a ledger, and uses ``interaction_tables.metrics`` (simple
    returns, from the starting NAV) -- the method that produced that row
    originally.

    The anchors differ by one day, and ``n_daily_returns`` shows it. A ledger
    row on the PM grid starts at the first PM close, which is already after that
    day's fills, so it has one return fewer than there are dates and its TR
    excludes the first day's move. The composite starts at the starting NAV
    and includes it.

WR, PLR
    Per position, the sum of ``realized`` over its opening fill and its closing
    (or expiry) fill: a round trip, net of the entry cost as well as the exit
    cost. Delta-hedge fills are excluded, and positions still open at the end of
    the window are not counted. ``WR_close_only`` / ``PLR_close_only`` count the
    closing fill alone, so they omit the entry cost; that is the definition
    ``interaction_tables.trades`` uses. It is written alongside for comparison,
    never in place of the round trip.

Validation -- the script refuses rather than prints a number
------------------------------------------------------------
* No manifest records a failure, and every run -- including each of the 140
  composite members -- opened at least one position. A baseline that never
  trades writes a clean, reconciling ledger and a return of exactly zero, so
  "it ran" is not evidence that it traded.
* Every row covers exactly the dates in ``--dates``.
* The environment configuration of every row is compared with the SOTA row key
  by key, and the keys that differ are written to the output.

Usage::

    python scripts/analysis/build_table1.py --runs "$PM_ARM_ROOT" \\
        --dates configs/dates_eval.txt \\
        --oracle "$PM_ARM_ROOT/oracle_S0.json" \\
        --out results/table1.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interaction_tables as it  # noqa: E402  -- produced the equal-weighted row

from portfolio_monkey.eval.builders.from_ledger import build_ledger_panel  # noqa: E402
from portfolio_monkey.eval.metrics import compute  # noqa: E402
from portfolio_monkey.eval.schema import NavPanel, NavRow, PanelProvenance  # noqa: E402

#: (row label, group, kind, default directory under ``--runs``)
ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("SOTA", "Language policy", "run", "c27_s4_none_step20"),
    ("Equal-weighted rule", "Rule-based", "composite", "baselines_pername_rlenv"),
    ("GARCH rule", "Rule-based", "run", "b9_garch_vrp_rlenv"),
    ("Threshold rule", "Rule-based", "run", "b9_econ_rules_rlenv"),
    ("GBDT", "Machine learning", "run", "pkgB_rlenv"),
    ("Logistic", "Machine learning", "run", "pkgLogit_rlenv"),
)
REFERENCE = "SOTA"

COLUMNS = (
    "row", "group",
    "TR_pct", "ASR", "ACR", "ASoR", "AVOL_pct", "MDD_pct", "WR_pct", "PLR",
    "n_round_trips", "WR_close_only_pct", "PLR_close_only",
    "first_date", "last_date", "n_daily_returns", "n_runs", "n_opens",
    "env_fingerprint", "mark_quote_source", "size_rule",
    "env_config_keys_differing_from_SOTA", "metric_path",
)


# --------------------------------------------------------------------------- io


def _fills(run: Path) -> Iterator[dict[str, Any]]:
    with (run / "fills.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for key, inner in value.items():
            out.update(_flatten(inner, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): json.dumps(value, sort_keys=True)}


def _manifest(run: Path) -> dict[str, Any]:
    path = run / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{run}: no manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("failure"):
        raise SystemExit(f"{run.name}: manifest records a failure: {manifest['failure']!r}")
    return manifest


def _opens(run: Path) -> int:
    return sum(1 for fill in _fills(run) if fill.get("kind") == "open")


# ---------------------------------------------------------------------- trades


def round_trips(runs: list[Path]) -> tuple[float | None, float | None, int]:
    """Win rate, profit/loss ratio and count of closed positions, entry cost included."""
    realized: dict[tuple[str, str], float] = defaultdict(float)
    closed: set[tuple[str, str]] = set()
    for run in runs:
        for fill in _fills(run):
            position = fill.get("position_id")
            if position in (None, "-") or fill.get("kind") == "hedge":
                continue
            key = (str(run), position)  # position ids are unique within a run only
            realized[key] += fill.get("realized") or 0.0
            if fill.get("kind") in ("close", "expire"):
                closed.add(key)
    values = [realized[key] for key in closed]
    wins = [v for v in values if v > 0]
    losses = [-v for v in values if v < 0]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided if decided else None
    ratio = (sum(wins) / len(wins)) / (sum(losses) / len(losses)) if wins and losses else None
    return win_rate, ratio, len(values)


# ------------------------------------------------------------------------ rows


def _environment(manifest: dict[str, Any]) -> dict[str, str]:
    env = _flatten(manifest["env_config"])
    marks = manifest.get("mark_quotes") or {}
    if marks.get("request_errors"):
        raise SystemExit(
            f"{manifest.get('arm')}: {marks['request_errors']} mark-quote request errors "
            f"(first: {marks.get('first_request_error')!r})"
        )
    env["__mark_quote_source"] = json.dumps(marks.get("source_version"))
    return env


def _check_dates(label: str, dates: list[dt.date], grid: list[dt.date]) -> None:
    if dates != grid:
        missing = sorted(set(grid) - set(dates))
        extra = sorted(set(dates) - set(grid))
        raise SystemExit(
            f"{label}: covers {dates[0] if dates else None} .. {dates[-1] if dates else None} "
            f"({len(dates)} dates), expected {grid[0]} .. {grid[-1]} ({len(grid)}); "
            f"missing {missing[:3]}, extra {extra[:3]}"
        )


def _ledger_metrics(m: Any) -> dict[str, float | None]:
    def pct(v: float | None) -> float | None:
        return None if v is None else 100.0 * v

    return {
        "TR_pct": pct(m.cumulative_return),
        "ASR": m.sharpe,
        "ACR": m.calmar,
        "ASoR": m.sortino,
        "AVOL_pct": pct(m.annualized_vol),
        "MDD_pct": pct(m.max_drawdown_daily),
    }


def single_run(label: str, run: Path, grid: list[dt.date]) -> tuple[dict, dict[str, str]]:
    manifest = _manifest(run)
    opens = _opens(run)
    if opens == 0:
        raise SystemExit(f"{label} ({run.name}): opened no position; refusing to report it")
    panel = build_ledger_panel(run, grid="PM")
    _check_dates(label, [r.trade_date for r in panel.pm_rows()], grid)
    metrics = compute(panel)
    win_rate, ratio, n = round_trips([run])
    wr_close, plr_close = it.trades([run])
    row = {
        **_ledger_metrics(metrics),
        "WR_pct": None if win_rate is None else 100.0 * win_rate,
        "PLR": ratio,
        "n_round_trips": n,
        "WR_close_only_pct": None if wr_close is None else 100.0 * wr_close,
        "PLR_close_only": plr_close,
        "n_daily_returns": metrics.n_obs,
        "n_runs": 1,
        "n_opens": opens,
        "env_fingerprint": manifest["env_fingerprint"],
        "metric_path": "build_ledger_panel(PM) -> eval.metrics.compute; WR/PLR round trip",
    }
    return row, _environment(manifest)


def composite(label: str, root: Path, grid: list[dt.date], tenor: str) -> tuple[dict, dict[str, str]]:
    arms, rejected = it.load(root, grid)
    if rejected:
        raise SystemExit(f"{label}: {len(rejected)} arms rejected by interaction_tables.load, "
                         f"first {rejected[:3]}")
    members = {key: value for key, value in arms.items() if key[2] == tenor}
    if not members:
        raise SystemExit(f"{label}: no arms at horizon {tenor!r} under {root}")
    runs = [path for _, path in members.values()]

    environments: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    fingerprints: set[str] = set()
    opens = 0
    for run in runs:
        manifest = _manifest(run)
        count = _opens(run)
        if count == 0:
            raise SystemExit(f"{label}: member {run.name} opened no position")
        opens += count
        fingerprints.add(manifest["env_fingerprint"])
        environments[tuple(sorted(_environment(manifest).items()))].append(run.name)
    if len(environments) != 1 or len(fingerprints) != 1:
        raise SystemExit(f"{label}: members ran under {len(environments)} environment "
                         f"configurations / {len(fingerprints)} fingerprints")

    levels = it.compose([returns for returns, _ in members.values()])
    m = it.metrics(levels)
    win_rate, ratio, n = round_trips(runs)
    wr_close, plr_close = it.trades(runs)
    row = {
        "TR_pct": 100.0 * m["tr"],
        "ASR": m["asr"],
        "ACR": m["acr"],
        "ASoR": m["asor"],
        "AVOL_pct": None if m["vol"] is None else 100.0 * m["vol"],
        "MDD_pct": 100.0 * m["mdd"],
        "WR_pct": None if win_rate is None else 100.0 * win_rate,
        "PLR": ratio,
        "n_round_trips": n,
        "WR_close_only_pct": None if wr_close is None else 100.0 * wr_close,
        "PLR_close_only": plr_close,
        "n_daily_returns": len(levels) - 1,
        "n_runs": len(runs),
        "n_opens": opens,
        "env_fingerprint": fingerprints.pop(),
        "metric_path": (f"interaction_tables composite, horizon {tenor}, daily-rebalanced "
                        "equal weight, simple returns from start NAV; WR/PLR round trip"),
    }
    return row, dict(next(iter(environments)))


def oracle(path: Path, grid: list[dt.date]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload["nav_curve"]
    rows = tuple(
        NavRow(
            arm="oracle",
            track="full",
            trade_date=dt.date.fromisoformat(p["trade_date"]),
            session=p["session"],
            nlv=float(p["nav"]),
            cash=float(p["cash"]),
            n_positions=int(p["open_positions"]),
        )
        for p in points
    )
    panel = NavPanel(
        arm="oracle",
        track="full",
        rows=rows,
        provenance=PanelProvenance(source="clairvoyant_oracle nav_curve", grid="PM"),
    )
    _check_dates("oracle", [r.trade_date for r in panel.pm_rows()], grid)
    metrics = compute(panel)
    sizing = payload.get("sizing") or {}
    return {
        **_ledger_metrics(metrics),
        "n_daily_returns": metrics.n_obs,
        "n_runs": 1,
        "n_opens": payload.get("trips_taken"),
        "env_fingerprint": "",
        "mark_quote_source": "",
        "size_rule": sizing.get("rule", ""),
        "env_config_keys_differing_from_SOTA": "n/a (not an OptionsEnv run)",
        "metric_path": ("nav_curve -> NavPanel -> eval.metrics.compute; WR/PLR undefined "
                        "for a hindsight schedule"),
    }


# ------------------------------------------------------------------------ main


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True,
                        help="directory holding the run directories named in ROWS")
    parser.add_argument("--dates", type=Path, required=True,
                        help="the test-window date file every row must cover exactly")
    parser.add_argument("--tenor", default="8_30",
                        help="contract horizon of the equal-weighted composite")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="clairvoyant_oracle.py output; adds the hindsight reference row")
    parser.add_argument("--run", action="append", default=[], metavar="ROW=DIR",
                        help="override the directory of one row")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    grid = sorted(dt.date.fromisoformat(line.strip())
                  for line in args.dates.read_text(encoding="utf-8").splitlines() if line.strip())
    overrides = dict(item.split("=", 1) for item in args.run)
    unknown = set(overrides) - {label for label, *_ in ROWS}
    if unknown:
        raise SystemExit(f"--run names unknown rows: {sorted(unknown)}")

    records: list[dict[str, Any]] = []
    environments: dict[str, dict[str, str]] = {}
    for label, group, kind, default in ROWS:
        path = Path(overrides.get(label, args.runs / default))
        if kind == "run":
            row, env = single_run(label, path, grid)
        else:
            row, env = composite(label, path, grid, args.tenor)
        row.update(row=label, group=group, first_date=grid[0].isoformat(),
                   last_date=grid[-1].isoformat())
        row["mark_quote_source"] = json.loads(env["__mark_quote_source"]) or ""
        row["size_rule"] = json.loads(env.get("size.size_rule", "null")) or ""
        environments[label] = env
        records.append(row)

    reference = environments[REFERENCE]
    for row in records:
        env = environments[row["row"]]
        differing = sorted(k for k in set(reference) | set(env) if reference.get(k) != env.get(k))
        row["env_config_keys_differing_from_SOTA"] = ";".join(differing)

    if args.oracle is not None:
        row = oracle(args.oracle, grid)
        row.update(row="Ex-post oracle (hindsight bound)", group="Reference",
                   first_date=grid[0].isoformat(), last_date=grid[-1].isoformat())
        records.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({column: _format(row.get(column)) for column in COLUMNS})

    head = (f"{'row':34s}{'TR%':>9}{'ASR':>8}{'ACR':>9}{'ASoR':>8}{'AVOL%':>8}"
            f"{'MDD%':>8}{'WR%':>8}{'PLR':>7}  differs from SOTA on")
    print(f"window {grid[0]} .. {grid[-1]} ({len(grid)} dates)\n{head}\n{'-' * len(head)}")
    for row in records:
        def f(key: str, spec: str = ".2f") -> str:
            value = row.get(key)
            return "--" if value is None else format(value, spec)

        print(f"{row['row']:34s}{f('TR_pct'):>9}{f('ASR'):>8}{f('ACR', ',.2f'):>9}"
              f"{f('ASoR'):>8}{f('AVOL_pct'):>8}{f('MDD_pct'):>8}{f('WR_pct'):>8}"
              f"{f('PLR'):>7}  {row['env_config_keys_differing_from_SOTA'] or '(reference)'}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
