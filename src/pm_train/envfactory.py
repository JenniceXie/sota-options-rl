"""The environment an arm is rolled out and evaluated in.

Built from ``paper_base_config`` plus the arm's ``env_overrides`` plus the mark
quote source, and nothing else.  The mark source is not an arm setting: every
teacher run in ``runs_main_sample/`` carries ``mark_quote_source =
massive_mark_quotes.v1`` in its fingerprint (``556df22b...``), and a rollout
without it marks unquoted legs at entry -- measured on 24 teacher runs as a
-2.8pp..+10.6pp change in realized return, mostly upward.  So it is on for every
arm, and :func:`arm_config` refuses to build a config whose fingerprint the base
arm cannot reproduce.
"""

from __future__ import annotations

import os
import threading
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from portfolio_monkey.env.chain import OptionChain
from portfolio_monkey.env.datasets import DatasetFeatureSource
from portfolio_monkey.env.environment import OptionsEnv, build_grid, monthly_episodes
from portfolio_monkey.env.ledger import LedgerWriter
from portfolio_monkey.env.markquotes import MARK_QUOTE_SOURCE_VERSION, MassiveMarkQuotes
from portfolio_monkey.training.spec import paper_base_config, paper_matrix

#: The fingerprint every shipped teacher run and the SFT corpus carry.  The
#: unablated arm must reproduce it exactly or it is not the same environment.
TEACHER_FINGERPRINT = "556df22b52af32d11a495ba87a8bc568566642c4897bbdc67cd4897ce363ebc5"

#: The only arms this campaign runs.  Anything else is refused by name.
IN_SCOPE = ("sys_sft", "sys_sft_rl", "s4_none")

#: Declared env_overrides keys that are not fields of the environment.  An arm
#: carrying one would silently run unablated under its ablation's name.
UNABLATABLE = frozenset({
    "a1_summary_a", "a1_no_shape_a", "a1_none_a",
    "s2_no_portfolio", "s3_no_history", "s4_headline", "s4_validated",
})


class ScopeError(ValueError):
    pass


def arm_overrides(arm_id: str) -> dict[str, Any]:
    if arm_id in UNABLATABLE:
        raise ScopeError(
            f"{arm_id}: declares an env_overrides key that is not a field of the "
            "environment; running it would execute the unablated env under the "
            "ablation's name. Permanently out of scope -- refused."
        )
    if arm_id not in IN_SCOPE:
        raise ScopeError(f"{arm_id}: not one of the three in-scope arms {IN_SCOPE}")
    arms = {a.arm_id: a for a in paper_matrix(seeds=(0,)).arms}
    return dict(arms[arm_id].env_overrides or {})


def arm_config(arm_id: str, max_context_tokens: int | None = None):
    """``max_context_tokens`` (or env PM_MAX_CONTEXT) widens the context budget.

    It only feeds the budget check -- the system block and grammar the policy
    sees are byte-identical -- but it is an EnvConfig field, so the fingerprint
    changes (32,768 -> 65,536: 556df22b... -> 88908866...).  The check below still
    requires the config to equal the teacher's in every other field.
    """
    if max_context_tokens is None and os.environ.get("PM_MAX_CONTEXT"):
        max_context_tokens = int(os.environ["PM_MAX_CONTEXT"])
    overrides = arm_overrides(arm_id)
    if max_context_tokens is not None and max_context_tokens != 32_768:
        cfg32 = paper_base_config(**{**overrides, "mark_quote_source": MARK_QUOTE_SOURCE_VERSION})
        if not overrides and cfg32.fingerprint() != TEACHER_FINGERPRINT:
            raise RuntimeError(f"{arm_id}: base (32k) config does not reproduce the teacher fingerprint")
        return paper_base_config(**{**overrides, "mark_quote_source": MARK_QUOTE_SOURCE_VERSION,
                                    "max_context_tokens": int(max_context_tokens)})
    config = paper_base_config(**{**overrides, "mark_quote_source": MARK_QUOTE_SOURCE_VERSION})
    base = paper_base_config(**{"mark_quote_source": MARK_QUOTE_SOURCE_VERSION})
    if base.fingerprint() != TEACHER_FINGERPRINT:
        raise RuntimeError(
            f"base config fingerprint {base.fingerprint()} != teacher {TEACHER_FINGERPRINT}; "
            "the rollout environment would not be the one the SFT corpus was generated in"
        )
    if not overrides and config.fingerprint() != TEACHER_FINGERPRINT:
        raise RuntimeError(f"{arm_id}: unablated arm does not reproduce the teacher fingerprint")
    return config


def read_dates(path: str | Path) -> list[date]:
    return [date.fromisoformat(line.strip()) for line in Path(path).read_text().split() if line.strip()]


def episodes_for(config, dates: list[date]):
    return monthly_episodes(build_grid(dates, config))


# -- per-process shared mark source ------------------------------------------

_MARKS: dict[str, MassiveMarkQuotes] = {}
_MARKS_LOCK = threading.Lock()


def mark_source(cache_dir: str | Path) -> MassiveMarkQuotes:
    """One mark source per process, with its own append-only cache file.

    ``MassiveMarkQuotes`` holds a lock, so threads in one process share it; a
    cache file per process avoids two processes interleaving appends.
    """
    key = str(cache_dir)
    with _MARKS_LOCK:
        if key not in _MARKS:
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            mine = cache / f"marks_{os.uname().nodename}_{os.getpid()}.jsonl"
            if not mine.exists():
                # Warm start from every other process's cache: the tape does not
                # change, so a quote (or a recorded miss) fetched once is valid
                # for every later rollout that holds the same contract.
                seen: set[str] = set()
                # A consolidated base.jsonl (see consolidate_marks) replaces the
                # scan: with one process per rollout the scan would re-read every
                # per-process file at every rollout start.
                base = cache / "base.jsonl"
                sources = [base] if base.exists() else sorted(cache.glob("*.jsonl"))
                with mine.open("w", encoding="utf-8") as out:
                    for other in sources:
                        if other == mine:
                            continue
                        for line in other.open(encoding="utf-8"):
                            if line.strip() and line not in seen:
                                seen.add(line)
                                out.write(line if line.endswith("\n") else line + "\n")
            timeout = float(os.getenv("MASSIVE_TIMEOUT_SECONDS", "30"))
            _MARKS[key] = MassiveMarkQuotes(cache_path=mine, timeout_seconds=timeout)
        return _MARKS[key]


def env_factory(config, *, data_root: str | Path, mark_cache_dir: str | Path,
                ledger: LedgerWriter | None = None):
    """A fresh ``OptionsEnv`` per rollout: two rollouts sharing one would trade
    against each other's book.  Chain and features are per-rollout too, because
    nothing in them is documented as thread-safe."""
    root = Path(data_root)

    def make():
        return OptionsEnv(
            config,
            chain=OptionChain(config.resolver, root=root),
            features=DatasetFeatureSource(root, extract_dir=None),
            ledger=ledger,
            marks=mark_source(mark_cache_dir),
        )

    return make


def consolidate_marks(cache_dir: str | Path) -> int:
    """Merge every cache file into base.jsonl (unique lines).  Run between jobs."""
    cache = Path(cache_dir)
    seen: set[str] = set()
    tmp = cache / "base.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as out:
        for f in sorted(cache.glob("*.jsonl")):
            for line in f.open(encoding="utf-8"):
                if line.strip() and line not in seen:
                    seen.add(line)
                    out.write(line if line.endswith("\n") else line + "\n")
    tmp.replace(cache / "base.jsonl")
    return len(seen)
