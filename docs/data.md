# Data requirements

Nothing in this repository is market data. This file states exactly what the code
expects, so a reader can judge the implementation and, if they hold equivalent
licences, supply their own tree.

## The environment datasets

Six market datasets and the news rows. `env/datasets.py` reads a partitioned
tree rooted at `$PM_DATA_ROOT`, laid out as
`features/<dataset>/date=<YYYY-MM-DD>/part-*.{jsonl,parquet}`.

| dataset | format | what the encoder takes from it |
| --- | --- | --- |
| `option_chain_snapshots` | **parquet only** | per-contract NBBO, greeks, open interest; the tradeable universe at each step |
| `option_iv_surface_features` | jsonl | `iv` level, `ts` term slope, `sk` 25-delta risk reversal, `bf` 25-delta butterfly |
| `option_delta_point_surface` | jsonl | implied volatility at a requested delta point, used for contract resolution |
| `option_open_interest_features` | jsonl | `doi`, the change in open interest |
| `compact_option_flow_states` | jsonl | `fi`, option flow imbalance |
| `underlying_market_features` | jsonl | underlying return and `rv` realized volatility |
| `news_catalyst_rows` | jsonl | the `N` changelog rows |

Field layouts: [`../examples/market_state_schema.json`](../examples/market_state_schema.json)
and [`../examples/news_schema.json`](../examples/news_schema.json).

`option_chain_snapshots` being parquet-only is why the `chain` extra exists. The
first chain read of the first step fails without `pyarrow`.

## Point-in-time discipline

Every field carries an availability timestamp. A state for date `t` is built only
from rows with `availability <= t`. The runner asserts this rather than trusting
it, and the assertion is what makes the held-out window held out.

One tolerated absence: a date with no `news_catalyst_rows` partition renders an
explicit unavailability row and continues. The five numeric datasets raise
instead — see the README for why.

## Provenance and redistribution

The underlying feeds are commercial: an options market-data vendor for chains and
flow, CRSP/WRDS for underlying quotes, SpiderRock for quote-at-print NBBO, TAQ for
session anchors, and a licensed news feed. **None of it may be redistributed**, so
none of it is here, and we cannot supply it on request.

Trading-date lists are the one exception: `configs/dates_{sft,rl,eval}.txt` are
NYSE session dates, which are public. They define the three windows —
63 / 59 / 124 dates, disjoint and chronologically ordered. They sit where the
code looks for them; `baselines/dataset.py` and several analysis scripts read
these exact paths.

## The supervised corpus

Not released. Each line of `train.jsonl` is one month-long episode:
`messages`, a per-message `step_loss_mask`, `metadata`, the `env_fingerprint`, the
teacher identifier, and the token accounting. Full schema in
[`../examples/sft_trajectory_schema.json`](../examples/sft_trajectory_schema.json);
a synthetic record in
[`../examples/synthetic_sft_example.jsonl`](../examples/synthetic_sft_example.jsonl).

Two details a compatible corpus must get right:

* **`step_loss_mask` is per message, and it is the field the trainer reads.** A
  top-level `loss_mask` alone is silently ignored by the trainer used here.
* **`env_fingerprint` must agree across every record**, and must match the
  environment the results are computed in. Otherwise the corpus and the results
  describe different environments.
