"""Lagged sequences per underlying, and a small LSTM over them.

``d_lstm`` is one of the paper's Band D arms, and the only one whose input shape
differs from the others.  The question it needs answered first is *what the
sequence is*, because there are two candidates and one of them is wrong:

* **the underlying's own market history** -- for a target date, the last ``T``
  daily market rows for that name.  This is a time series and is what an LSTM is
  for.
* the ten names at one date.  This is a cross-section, not a sequence: its order
  is arbitrary, so a recurrent model over it would be learning the alphabet.

The first is used.

**The lookback crosses window boundaries, deliberately and non-leakily.**  The
three feature runs cover 63 + 59 + 124 = 246 contiguous trading dates, so a
sequence targeting the first eval date reads market rows from the RL window.
Those are *past* observations, available in real time at that decision, so this
is not look-ahead -- but it does mean the eval rows are not independent of the
validation window's inputs, and the split is on the **target** date only.  Said
here because "my test set touched training dates" is the reasonable first
suspicion and the answer is that it touched their past, which every real
deployment also does.

**What it costs.**  The first ``T - 1`` dates of the whole history have no full
lookback and are dropped rather than padded -- padding would invent market rows
and the model would learn the padding.  Since SFT is the first window, the cost
falls entirely on training: at ``T = 10``, 61 usable train dates become 52.

**The honest expectation.**  A single LSTM layer with 16 hidden units over 20
features is about 2,400 parameters, against roughly 610 independent
``(date, underlying)`` cells -- the four tenors of a package share one feature
vector, so the row count overstates independence fourfold.  This is
overparameterized by construction, and the permutation control is what will say
whether anything survives it.  It is implemented because the matrix declares the
arm, not because the sample supports it.

**The sequence is flattened into the feature vector** (``T * F`` columns, then
the tenor index last) so that every consumer -- ``packages.evaluate``, the
threshold fit, the policy -- keeps working unchanged, and the recurrent model
reshapes internally.  That also makes the MLP directly comparable on identical
inputs rather than on a different view of them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import Frame
from .packages import TENOR_INDEX, PackageSet

#: Trading days of history per sequence, target date included.
LOOKBACK = 10


@dataclass(frozen=True, slots=True)
class _Series:
    """Date-ordered market rows for one underlying, across every window."""

    dates: tuple[str, ...]
    rows: np.ndarray  # (n_dates, n_features)


def _series(frames: Sequence[Frame]) -> tuple[dict[str, _Series], tuple[str, ...]]:
    names = sorted({r.underlying for f in frames for r in f.rows})
    feature_names = frames[0].feature_names
    for f in frames[1:]:
        if f.feature_names != feature_names:
            raise ValueError(
                "feature frames disagree on their columns; concatenating them "
                "would align a skew column against a term-slope column"
            )
    out: dict[str, _Series] = {}
    for name in names:
        pairs = sorted(
            (
                (r.trade_date, [r.features[c] for c in feature_names])
                for f in frames
                for r in f.rows
                if r.underlying == name
            ),
            key=lambda kv: kv[0],
        )
        dedup: dict[str, list] = {}
        for day, vec in pairs:
            dedup[day] = vec  # a date seen twice is one date
        days = tuple(sorted(dedup))
        out[name] = _Series(
            dates=days,
            rows=np.array(
                [[np.nan if v is None else float(v) for v in dedup[d]] for d in days],
                dtype=float,
            ),
        )
    return out, feature_names


def build(
    dump: Path,
    frames: Sequence[Frame],
    *,
    target_dates: Sequence[str],
    lookback: int = LOOKBACK,
    min_rows: int = 200,
) -> PackageSet:
    """One binary problem per head, with a flattened ``lookback`` window of state.

    ``frames`` should be every feature frame available, in any order -- the
    series are rebuilt date-sorted.  ``target_dates`` is the split: a row is kept
    only if its target date is in it, while its lookback may reach earlier.
    """
    series, feature_names = _series(frames)
    index = {
        name: {d: i for i, d in enumerate(s.dates)} for name, s in series.items()
    }
    wanted = set(target_dates)
    acc: dict[str, list[tuple[np.ndarray, int, str]]] = {}
    with dump.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            day, name = row["trade_date"], row["underlying"]
            if day not in wanted or name not in index:
                continue
            pos = index[name].get(day)
            if pos is None or pos + 1 < lookback:
                continue  # no full history; not padded
            tenor = TENOR_INDEX.get(row["tenor"])
            if tenor is None:
                continue
            window = series[name].rows[pos + 1 - lookback : pos + 1]
            acc.setdefault(row["head"], []).append(
                (
                    np.concatenate([window.reshape(-1), [float(tenor)]]),
                    1 if row["profit"] > 0 else 0,
                    day,
                )
            )
    heads = tuple(
        h for h in sorted(acc)
        if len(acc[h]) >= min_rows and len({lab for _, lab, _ in acc[h]}) == 2
    )
    flat_names = tuple(
        f"{c}_lag{lookback - 1 - t}"
        for t in range(lookback)
        for c in feature_names
    ) + ("tenor_index",)
    return PackageSet(
        heads=heads,
        X={h: np.stack([r[0] for r in acc[h]]) for h in heads},
        y={h: np.array([r[1] for r in acc[h]], dtype=int) for h in heads},
        dates={h: np.array([r[2] for r in acc[h]], dtype=object) for h in heads},
        feature_names=flat_names,
    )


class LSTMClassifier:
    """A small recurrent classifier with an sklearn-shaped interface.

    ``fit``/``predict_proba`` take the flattened 2D array ``build`` produces and
    reshape internally, so this drops into ``packages.evaluate`` and the
    threshold fit without either of them knowing it is recurrent.
    """

    def __init__(
        self,
        *,
        n_features: int,
        lookback: int = LOOKBACK,
        hidden: int = 16,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 256,
        seed: int = 0,
        patience: int = 10,
        val_fraction: float = 0.15,
    ) -> None:
        self.n_features = n_features
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.patience = patience
        self.val_fraction = val_fraction
        self._net = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # -- shaping ---------------------------------------------------------

    def _split_inputs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq, tenor = X[:, :-1], X[:, -1:]
        expected = self.lookback * self.n_features
        if seq.shape[1] != expected:
            raise ValueError(
                f"expected {expected} sequence columns for lookback="
                f"{self.lookback} x {self.n_features} features, got {seq.shape[1]}"
            )
        return seq.reshape(len(X), self.lookback, self.n_features), tenor

    def _standardize(self, seq: np.ndarray, *, fit: bool) -> np.ndarray:
        flat = seq.reshape(-1, self.n_features)
        if fit:
            self._mean = np.nanmean(flat, axis=0)
            self._std = np.nanstd(flat, axis=0)
            self._std[self._std == 0] = 1.0
        assert self._mean is not None and self._std is not None
        out = (seq - self._mean) / self._std
        # Imputation after centring, so a missing cell becomes the training
        # mean rather than a zero that happens to mean something else.
        return np.nan_to_num(out, nan=0.0)

    # -- fitting ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        seq, tenor = self._split_inputs(np.asarray(X, dtype=float))
        seq = self._standardize(seq, fit=True)
        y = np.asarray(y, dtype=float)

        n_val = max(1, int(len(y) * self.val_fraction))
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(y))
        val_idx, tr_idx = order[:n_val], order[n_val:]

        class Net(nn.Module):
            def __init__(self, n_features: int, hidden: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
                self.head = nn.Linear(hidden + 1, 1)

            def forward(self, s, t):
                _, (h, _) = self.lstm(s)
                return self.head(torch.cat([h[-1], t], dim=1)).squeeze(1)

        net = Net(self.n_features, self.hidden)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Positive-class weighting, because base rates here run from 0.10 to
        # 0.84 and an unweighted fit on bf collapses to the constant.
        pos = float(y[tr_idx].mean())
        pos_weight = torch.tensor([(1 - pos) / pos] if 0 < pos < 1 else [1.0])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        S = torch.tensor(seq, dtype=torch.float32)
        T = torch.tensor(tenor, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        best, best_state, waited = float("inf"), None, 0
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(tr_idx))
            for i in range(0, len(tr_idx), self.batch_size):
                b = torch.tensor(tr_idx[perm[i : i + self.batch_size].numpy()])
                opt.zero_grad()
                loss_fn(net(S[b], T[b]), Y[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = torch.tensor(val_idx)
                vl = float(loss_fn(net(S[v], T[v]), Y[v]))
            if vl < best - 1e-5:
                best, waited = vl, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("fit before predict_proba")
        seq, tenor = self._split_inputs(np.asarray(X, dtype=float))
        seq = self._standardize(seq, fit=False)
        with torch.no_grad():
            logit = self._net(
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(tenor, dtype=torch.float32),
            )
            p = torch.sigmoid(logit).numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
