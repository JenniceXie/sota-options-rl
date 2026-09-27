"""Feature builders for research datasets."""

from portfolio_monkey.data.features.candidates import (
    CandidateFilterConfig,
    evaluate_strategy_package,
    generate_candidate_rows,
    generate_strategy_template_candidates,
    passes_candidate_filters,
)
from portfolio_monkey.data.features.strategy_templates import (
    TEMPLATE_SPECS,
    canonical_underlying,
    parse_strategy_handle,
    strategy_handle,
    template_legs,
)
from portfolio_monkey.data.features.context import (
    StockContextFeatureBuilder,
    context_evidence_handle,
    stock_key,
)
from portfolio_monkey.data.features.daily import AdaptiveDailyFeatureBuilder
from portfolio_monkey.data.features.flow import (
    ClassifiedOptionTrade,
    CompactOptionFlowStateBuilder,
    QuoteRuleClassifier,
    TradeDirection,
    is_opra_auction,
)

__all__ = [
    "AdaptiveDailyFeatureBuilder",
    "CandidateFilterConfig",
    "ClassifiedOptionTrade",
    "CompactOptionFlowStateBuilder",
    "QuoteRuleClassifier",
    "StockContextFeatureBuilder",
    "TEMPLATE_SPECS",
    "TradeDirection",
    "context_evidence_handle",
    "canonical_underlying",
    "evaluate_strategy_package",
    "generate_candidate_rows",
    "generate_strategy_template_candidates",
    "is_opra_auction",
    "passes_candidate_filters",
    "parse_strategy_handle",
    "strategy_handle",
    "stock_key",
    "template_legs",
]
