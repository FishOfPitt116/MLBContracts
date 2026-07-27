"""Phase 0 contract predictor, packaged so it can be used as a sub-agent tool.

Re-exports the names agent/backtest.py (and anything else outside this
package) previously imported from the flat agent/predict.py module.
"""

from agent.predict.predict import actual_aav_for, lookup_player, predict_for, run_prediction

__all__ = ["actual_aav_for", "lookup_player", "predict_for", "run_prediction"]
