# Data

The inputs the code reads: their layout, their point-in-time rule and their
sources.

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
instead, because a plausible `na` in a field the policy reads as a measurement is
worse than a halt.

## Sources

The datasets are proprietary. Option trades and quotes come from OPRA, together
with OptionMetrics. Underlying prices are CRSP daily open and close prices for the
nine equities, and SpiderRock underlying marks for SPY. News comes from
Massive.com's news API, and regulatory filings from SEC EDGAR.

The trading-date lists `configs/dates_{sft,rl,eval}.txt` are NYSE session dates.
They define the three windows (63 / 59 / 124 dates, disjoint and in
chronological order). They sit where the code looks for them:
`baselines/dataset.py` and several analysis scripts read these exact paths.

## The supervised corpus

Each line of `train.jsonl` is one month-long episode:
`messages`, a per-message `step_loss_mask`, `metadata`, the `env_fingerprint`, the
teacher identifier, and the token accounting. Full schema in
[`../examples/sft_trajectory_schema.json`](../examples/sft_trajectory_schema.json);
a synthetic record in
[`../examples/synthetic_sft_example.jsonl`](../examples/synthetic_sft_example.jsonl).

Two details of the format matter:

* **`step_loss_mask` is per message, and it is the field the trainer reads.** A
  top-level `loss_mask` alone is silently ignored by the trainer used here.
* **`env_fingerprint` must agree across every record**, and must match the
  environment the results are computed in. Otherwise the corpus and the results
  describe different environments.
