"""The environment in its own process, one per rollout.

WHY.  ``RolloutSession`` runs ``run_arm`` on a worker *thread*.  veRL 0.9.1
dispatches a batch by prompt (``prompts.chunk(n_workers)``) and expands each
prompt into its G samples inside one worker, so all G=8 rollouts of a month run
as 8 environment threads in one process and serialise on the GIL: measured
env time per rollout 120-148 s under 0.9.1 vs 20-26 s under 0.7.1, whose
dispatch put one rollout per process.  The environment is deterministic given
the actions, so which process it runs in changes timing only; that is checked
byte-for-byte by ``tests/test_corpus_equivalence.py`` in process mode.

The child owns the ledger, the environment and the ``RolloutSession`` (so the
per-episode ``EpisodicQueuePolicy`` behaviour is unchanged); the parent only
exchanges ``Turn`` objects and completion text over a pipe.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any, Sequence


def _child(conn, spec: dict) -> None:
    ledger = None
    try:
        from portfolio_monkey.env.book import BookState  # noqa: F401  (pickle targets)
        from portfolio_monkey.env.ledger import LedgerWriter
        from portfolio_monkey.training.agent import RolloutSession

        from .envfactory import arm_config, env_factory, episodes_for, read_dates
        from .episode import EpisodicQueuePolicy, _run_result

        config = arm_config(spec["arm_id"])
        by_id = {e.episode_id: e for e in episodes_for(config, read_dates(spec["dates_file"]))}
        episodes = [by_id[i] for i in spec["episode_ids"]]
        run_dir = Path(spec["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger = LedgerWriter(run_dir, arm=spec["label"], track="full")
        run_kwargs: dict[str, Any] = {}
        if spec.get("manifest_extra"):
            run_kwargs["manifest_extra"] = spec["manifest_extra"]
        session = RolloutSession(
            env_factory(config, data_root=spec["data_root"],
                        mark_cache_dir=spec["mark_cache_dir"], ledger=ledger),
            episodes, arm=spec["label"], ledger_factory=lambda: ledger,
            state_dir=run_dir / "state", timeout=spec["timeout"], run_kwargs=run_kwargs)
        session._policy = EpisodicQueuePolicy(session._out, session._in, spec["timeout"])
        turn = session.start()
        while True:
            conn.send(("turn", turn))
            if turn is None:
                break
            kind, payload = conn.recv()
            if kind == "close":
                session.close()
                conn.send(("closed", None))
                return
            turn = session.respond(payload)
        run = _run_result(session)
        conn.send(("result", {"log_return": float(run.log_return), "start_nav": float(run.start_nav),
                              "end_nav": float(run.end_nav)}))
    except BaseException:  # noqa: BLE001 - reported to the parent
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if ledger is not None:
            ledger.close()


class ProcessRolloutSession:
    """``start`` / ``respond`` / ``close`` / ``result`` over a pipe."""

    remote = True

    def __init__(self, *, arm_id: str, label: str, episode_ids: Sequence[str], dates_file: str,
                 run_dir: str | Path, data_root: str | Path, mark_cache_dir: str | Path,
                 timeout: float = 1800.0, manifest_extra: dict | None = None):
        self._spec = {
            "arm_id": arm_id, "label": label, "episode_ids": list(episode_ids),
            "dates_file": str(dates_file), "run_dir": str(run_dir), "data_root": str(data_root),
            "mark_cache_dir": str(mark_cache_dir), "timeout": float(timeout),
            "manifest_extra": manifest_extra,
        }
        self._timeout = float(timeout)
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_child, args=(child, self._spec), daemon=False)
        self._result: dict | None = None

    def _recv(self):
        if not self._conn.poll(self._timeout):
            self.close()
            raise RuntimeError(f"{self._spec['label']}: no message from the env process in {self._timeout}s")
        kind, payload = self._conn.recv()
        if kind == "error":
            self._proc.join(timeout=30)
            raise RuntimeError(f"{self._spec['label']}: environment process failed:\n{payload}")
        return kind, payload

    def _next_turn(self):
        kind, payload = self._recv()
        assert kind == "turn", kind
        if payload is None:
            kind, self._result = self._recv()
            assert kind == "result", kind
            self._proc.join(timeout=60)
        return payload

    def start(self):
        self._proc.start()
        return self._next_turn()

    def respond(self, text: str):
        self._conn.send(("text", text))
        return self._next_turn()

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._conn.send(("close", None))
                if self._conn.poll(30):
                    self._conn.recv()
            except Exception:
                pass
            self._proc.join(timeout=30)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=10)

    def result(self) -> dict:
        if self._result is None:
            raise RuntimeError("episode has not ended")
        return self._result
