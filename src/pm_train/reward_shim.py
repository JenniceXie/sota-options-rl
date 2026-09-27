"""veRL loads custom_reward_function.path with spec_from_file_location and does
not register the module in sys.modules, which breaks @dataclass in
portfolio_monkey/training/reward.py ('NoneType' has no attribute '__dict__').
This file has no dataclasses; it imports the real function normally."""
from portfolio_monkey.training.reward import compute_score  # noqa: F401
