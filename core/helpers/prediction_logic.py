# core/helpers/prediction_logic.py
"""
Helper functions related to prediction interpretation, signal generation,
trade level calculation, and similarity analysis.
These functions are primarily used by the prediction modules.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# --- Constants for Signal Strength and Confidence (can be moved to config.yaml) ---
# These are illustrative values and should be tuned.

# For get_signal_strength:
DELTA_THRESHOLD_STRONG = 0.025  # e.g., 2.5% expected move for strong signal consideration
DELTA_THRESHOLD_MODERATE = 0.010  # e.g., 1.0% expected move for moderate signal consideration
CONFIDENCE_THRESHOLD_STRONG = 0.15  # e.g., P(best) - P(second_best) > 0.15
CONFIDENCE_THRESHOLD_MODERATE = 0.05
SIGMA_HIST_RELIABLE = 0.010  # Historical std dev < 1% considered reliable pattern
SIGMA_HIST_UNRELIABLE = 0.020  # Historical std dev > 2% considered unreliable pattern

# For get_confidence_hint:
CONF_HINT_VERY_HIGH = 0.20
CONF_HINT_HIGH = 0.10
CONF_HINT_MODERATE = 0.05
CONF_HINT_LOW_CAUTION = 0.02  # Threshold below which signals might be skipped

# For is_conflict:
CONFLICT_DELTA_HISTORY_IGNORE_THRESHOLD = 0.005  # If |delta_history| is less than this, don't flag conflict

# Default Risk/Reward for calculate_trade_levels
DEFAULT_RR_RATIO = 2.0


def compute_final_delta(
        delta_model,
        delta_history_avg,
        sigma_history_std,
        min_sigma_thresh=SIGMA_HIST_RELIABLE,  # Parameterize thresholds
        max_sigma_thresh=SIGMA_HIST_UNRELIABLE,
        weight_hist_at_min_sigma=0.6,
        weight_hist_at_max_sigma=0.2
):
    """
    Computes a final delta by blending model prediction with historical similar deltas.
    The weight given to historical data depends on the stability (std dev) of similar historical deltas.

    Args:
        delta_model (float): Delta predicted by the regression model.
        delta_history_avg (float): Average delta from similar historical situations.
        sigma_history_std (float): Standard deviation of deltas from similar historical situations.
        min_sigma_thresh (float): Sigma below which history gets max weight.
        max_sigma_thresh (float): Sigma above which history gets min weight.
        weight_hist_at_min_sigma (float): Weight for history when sigma is at/below min_sigma_thresh.
        weight_hist_at_max_sigma (float): Weight for history when sigma is at/above max_sigma_thresh.

    Returns:
        float: The blended final delta, rounded to 5 decimal places.
    """
    if pd.isna(delta_model):  # If model delta is NaN, can't compute final delta reliably
        logger.debug("Model delta is NaN, cannot compute final delta.")
        return np.nan

    if pd.isna(delta_history_avg) or pd.isna(sigma_history_std):
        logger.debug("Historical delta average or sigma is NaN. Using model delta directly.")
        return round(delta_model, 5)

    # Ensure sigma_history_std is not too small to avoid division by zero or instability if used in inverse weighting
    # For linear interpolation of weights, this check is less critical but good practice.
    if sigma_history_std < 1e-9:
        sigma_history_std = 1e-9
        # Alternative: if sigma_history_std is near zero, history is very consistent.
        # Could give full weight to history or model depending on philosophy.
        # For now, let the interpolation handle it.

    w_hist = 0.0  # Default weight for history if outside defined sigma ranges (or if logic fails)
    if sigma_history_std <= min_sigma_thresh:
        w_hist = weight_hist_at_min_sigma
    elif sigma_history_std >= max_sigma_thresh:
        w_hist = weight_hist_at_max_sigma
    else:
        # Linear interpolation of history weight
        # As sigma_history_std goes from min_sigma_thresh to max_sigma_thresh,
        # w_hist goes from weight_hist_at_min_sigma down to weight_hist_at_max_sigma.
        alpha = (sigma_history_std - min_sigma_thresh) / (max_sigma_thresh - min_sigma_thresh)
        w_hist = weight_hist_at_min_sigma - alpha * (weight_hist_at_min_sigma - weight_hist_at_max_sigma)

    w_model = 1.0 - w_hist

    final_delta = w_model * delta_model + w_hist * delta_history_avg
    return round(final_delta, 5)


def get_signal_strength(delta_final, confidence_score, sigma_history_std):
    """
    Determines the strength of a trading signal based on final delta, model confidence,
    and historical pattern stability.

    Args:
        delta_final (float): The final blended delta.
        confidence_score (float): Confidence from the classification model (e.g., P(best) - P(second_best)).
        sigma_history_std (float): Standard deviation of deltas from similar historical situations.

    Returns:
        str: Signal strength ("🟢 Сильный", "🟡 Умеренный", "⚪ Слабый").
    """
    if pd.isna(delta_final) or pd.isna(confidence_score):
        return "⚪ Слабый (Нет данных)"  # Or "N/A"

    if pd.isna(sigma_history_std):  # If no historical data, treat as unreliable
        sigma_history_std = SIGMA_HIST_UNRELIABLE + 0.01  # Ensure it's considered unreliable

    abs_delta = abs(delta_final)

    # Check for strong signal candidates first
    if abs_delta > DELTA_THRESHOLD_STRONG and confidence_score > CONFIDENCE_THRESHOLD_STRONG:
        if sigma_history_std < SIGMA_HIST_RELIABLE:
            return "🟢 Сильный"
        elif sigma_history_std < SIGMA_HIST_UNRELIABLE:
            return "🟡 Умеренный"  # Downgrade due to moderately reliable history
        else:
            return "⚪ Слабый"  # Downgrade significantly due to unreliable history
    # Check for moderate signal candidates
    elif abs_delta > DELTA_THRESHOLD_MODERATE and confidence_score > CONFIDENCE_THRESHOLD_MODERATE:
        if sigma_history_std < SIGMA_HIST_RELIABLE:
            # Could be upgraded, but let's keep it moderate for safety
            # Or, if delta is also high, could be "Сильный (менее уверенный)"
            return "🟡 Умеренный"
        elif sigma_history_std < SIGMA_HIST_UNRELIABLE:
            return "🟡 Умеренный"
        else:
            return "⚪ Слабый"  # Downgrade due to unreliable history
    # Otherwise, it's a weak signal
    else:
        return "⚪ Слабый"


def is_conflict(delta_model, delta_history_avg):
    """
    Checks if there's a conflict between model's delta prediction and historical average delta.
    A conflict is when they have opposite signs and historical delta is not negligible.

    Args:
        delta_model (float): Delta predicted by the regression model.
        delta_history_avg (float): Average delta from similar historical situations.

    Returns:
        bool: True if a conflict exists, False otherwise.
    """
    if pd.isna(delta_model) or pd.isna(delta_history_avg):
        return False  # No conflict if one of the values is missing

    # Only consider it a conflict if the historical delta itself is somewhat significant
    if abs(delta_history_avg) < CONFLICT_DELTA_HISTORY_IGNORE_THRESHOLD:
        return False

    # Conflict if signs are opposite
    return (delta_model > 0 and delta_history_avg < 0) or \
        (delta_model < 0 and delta_history_avg > 0)


def get_confidence_hint(score):
    """
    Provides a textual hint based on the confidence score.

    Args:
        score (float): The confidence score (e.g., P(best) - P(second_best)).

    Returns:
        str: A textual hint about the confidence level.
    """
    if pd.isna(score):
        return "Уверенность N/A"
    if score > CONF_HINT_VERY_HIGH:
        return "Очень высокая уверенность"
    elif score > CONF_HINT_HIGH:
        return "Высокая уверенность"
    elif score > CONF_HINT_MODERATE:
        return "Умеренная уверенность"
    elif score > CONF_HINT_LOW_CAUTION:
        return "Низкая уверенность - осторожно"
    else:
        return "Очень низкая уверенность - пропустить"


def calculate_trade_levels(entry_price, direction, predicted_volatility, risk_reward_ratio=DEFAULT_RR_RATIO):
    """
    Calculates Stop Loss (SL) and Take Profit (TP) levels.
    SL is based on predicted_volatility (interpreted as an ATR-like measure or expected move).
    TP is based on SL and risk_reward_ratio.

    Args:
        entry_price (float): The entry price of the trade.
        direction (str): Trade direction, 'long' or 'short'.
        predicted_volatility (float): Predicted volatility or expected move size (e.g., from reg_vol model).
                                      This is used as the basis for SL distance.
        risk_reward_ratio (float, optional): Desired risk/reward ratio for TP. Defaults to DEFAULT_RR_RATIO.

    Returns:
        tuple: (stop_loss, take_profit). Returns (np.nan, np.nan) if inputs are invalid.
    """
    if pd.isna(entry_price) or pd.isna(predicted_volatility) or predicted_volatility <= 1e-9:
        logger.debug("Invalid inputs for trade level calculation (NaN or non-positive volatility).")
        return np.nan, np.nan

    # Ensure volatility is positive if it's extremely small but negative
    # (models occasionally might predict tiny negative values for volatility if not constrained)
    actual_vol_for_sl = abs(predicted_volatility) if predicted_volatility < 0 else predicted_volatility
    if actual_vol_for_sl <= 1e-9:  # Rec-check after abs
        logger.debug(f"Predicted volatility ({predicted_volatility}) is too small for SL calculation.")
        return np.nan, np.nan

    sl_distance = actual_vol_for_sl * entry_price  # SL distance as percentage of entry price
    tp_distance = sl_distance * risk_reward_ratio

    sl, tp = np.nan, np.nan

    if direction.lower() == 'long':
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    elif direction.lower() == 'short':
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    else:
        logger.warning(f"Invalid trade direction for level calculation: {direction}")
        return np.nan, np.nan

    # Rounding: Determine a reasonable number of decimal places based on entry price
    # This is a heuristic. For very low-priced assets, more precision is needed.
    if entry_price > 1000:
        rounding_decimals = 2
    elif entry_price > 10:
        rounding_decimals = 4
    elif entry_price > 0.1:
        rounding_decimals = 5
    else:
        rounding_decimals = 6  # Or even more for sub-penny assets

    sl = round(max(0, sl), rounding_decimals)  # SL/TP cannot be negative
    tp = round(max(0, tp), rounding_decimals)

    return sl, tp


def similarity_analysis(
        x_live_features,
        df_historical_features,
        series_historical_deltas,
        top_n_similar=15,
        min_historical_samples=30  # Min samples needed in history to attempt analysis
):
    """
    Performs a cosine similarity analysis to find historical situations similar
    to the current live features and analyzes their subsequent deltas.

    Args:
        x_live_features (pd.DataFrame): DataFrame with a single row of live features.
        df_historical_features (pd.DataFrame): DataFrame of historical features.
                                               Must have the same columns as x_live_features.
        series_historical_deltas (pd.Series): Series of historical deltas, indexed Aligning
                                              with df_historical_features.
        top_n_similar (int): Number of most similar historical points to consider.
        min_historical_samples (int): Minimum number of historical samples required after cleaning.

    Returns:
        tuple: (avg_similar_delta, std_similar_delta, hint_string)
               Returns (np.nan, np.nan, "Error/Insufficient Data") on failure.
    """
    if df_historical_features.empty or len(df_historical_features) < min_historical_samples:
        msg = f"Insufficient historical data (have {len(df_historical_features)}, need {min_historical_samples}) for similarity analysis."
        logger.debug(msg)
        return np.nan, np.nan, msg

    if not x_live_features.columns.equals(df_historical_features.columns):
        common_cols = x_live_features.columns.intersection(df_historical_features.columns)
        if len(common_cols) < 1:  # Need at least one common feature
            msg = "No common features between live and historical data for similarity analysis."
            logger.error(msg)
            return np.nan, np.nan, msg
        logger.warning(f"Live and historical features columns mismatch. Using {len(common_cols)} common columns for similarity.")
        x_live_common = x_live_features[common_cols]
        df_historical_common = df_historical_features[common_cols]
    else:
        x_live_common = x_live_features
        df_historical_common = df_historical_features

    # Align historical deltas with historical features and clean NaNs
    # Ensure series_historical_deltas has the same index as df_historical_common
    if not series_historical_deltas.index.equals(df_historical_common.index):
        try:
            aligned_deltas = series_historical_deltas.reindex(df_historical_common.index)
        except Exception as e:
            msg = f"Error aligning historical deltas index: {e}"
            logger.error(msg)
            return np.nan, np.nan, msg
    else:
        aligned_deltas = series_historical_deltas

    # Drop rows where ANY feature is NaN in historical data or where delta is NaN
    valid_indices_features = df_historical_common.dropna().index
    valid_indices_deltas = aligned_deltas.dropna().index
    common_valid_indices = valid_indices_features.intersection(valid_indices_deltas)

    if len(common_valid_indices) < min_historical_samples:
        msg = f"Insufficient clean historical data after NaN removal (have {len(common_valid_indices)}, need {min_historical_samples})."
        logger.debug(msg)
        return np.nan, np.nan, msg

    df_hist_clean = df_historical_common.loc[common_valid_indices]
    series_deltas_clean = aligned_deltas.loc[common_valid_indices]

    # Also check x_live_common for NaNs
    if x_live_common.isnull().values.any():
        nan_live_features = x_live_common.columns[x_live_common.isnull().any()].tolist()
        msg = f"Live features contain NaNs: {nan_live_features}. Cannot perform similarity."
        logger.warning(msg)
        return np.nan, np.nan, msg

    # Scale features
    scaler = StandardScaler()
    try:
        # Fit scaler ONLY on historical data, then transform both historical and live data
        x_hist_scaled = scaler.fit_transform(df_hist_clean)
        x_live_scaled = scaler.transform(x_live_common)  # x_live_common should be a 2D array (1 row)
    except ValueError as e:  # Can happen if a column has zero variance after cleaning
        msg = f"Error during StandardScaler fitting/transforming: {e}. Check for zero-variance features."
        logger.error(msg)
        return np.nan, np.nan, msg
    except Exception as e:
        msg = f"Unexpected error during scaling: {e}"
        logger.error(msg, exc_info=True)
        return np.nan, np.nan, msg

    # Calculate cosine similarities
    similarities = cosine_similarity(x_hist_scaled, x_live_scaled).flatten()

    # Get top N similar indices (from the cleaned historical data)
    # Ensure top_n is not greater than available similarities
    actual_top_n = min(top_n_similar, len(similarities))
    if actual_top_n <= 0:
        msg = "No similarities calculated or actual_top_n is zero."
        logger.debug(msg)
        return np.nan, np.nan, msg

    top_indices_in_cleaned = similarities.argsort()[-actual_top_n:][::-1]  # Indices within x_hist_scaled/df_hist_clean

    # Get the actual deltas for these top similar historical points
    deltas_of_similar = series_deltas_clean.iloc[top_indices_in_cleaned]

    if deltas_of_similar.empty:
        msg = "No deltas found for the top similar historical points."
        logger.debug(msg)
        return np.nan, np.nan, msg

    avg_similar_delta = deltas_of_similar.mean()
    std_similar_delta = deltas_of_similar.std()

    # Handle case where std is NaN (e.g., if actual_top_n is 1, std is NaN, should be 0)
    if pd.isna(std_similar_delta) and actual_top_n == 1:
        std_similar_delta = 0.0
    elif pd.isna(std_similar_delta):  # If still NaN for other reasons
        logger.warning("std_similar_delta is NaN, but more than 1 similar point. This is unexpected.")

    # Provide a hint about the stability of these historical deltas
    if pd.isna(avg_similar_delta) or pd.isna(std_similar_delta):
        hint = "Не удалось рассчитать статистику схожести"
    elif std_similar_delta > SIGMA_HIST_UNRELIABLE:  # Use the same thresholds as signal strength
        hint = "Высокий разброс (нестабильный паттерн)"
    elif std_similar_delta > SIGMA_HIST_RELIABLE:
        hint = "Умеренный разброс (умеренно стабильно)"
    else:  # std_similar_delta <= SIGMA_HIST_RELIABLE
        hint = "Низкий разброс (стабильный паттерн)"

    return round(avg_similar_delta, 5), round(std_similar_delta, 5), hint


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    print("\n--- Testing compute_final_delta ---")
    print(f"Model=0.01, HistAvg=0.005, HistStd=0.005 -> Final: {compute_final_delta(0.01, 0.005, 0.005)}")  # High weight to hist
    print(f"Model=0.01, HistAvg=0.005, HistStd=0.010 -> Final: {compute_final_delta(0.01, 0.005, 0.010)}")  # Mid weight
    print(f"Model=0.01, HistAvg=0.005, HistStd=0.020 -> Final: {compute_final_delta(0.01, 0.005, 0.020)}")  # Low weight to hist
    print(f"Model=0.01, HistAvg=NaN, HistStd=0.010 -> Final: {compute_final_delta(0.01, np.nan, 0.010)}")  # Use model

    print("\n--- Testing get_signal_strength ---")
    print(f"Delta=0.03, Conf=0.2, HistStd=0.005 -> Strength: {get_signal_strength(0.03, 0.2, 0.005)}")  # Strong
    print(f"Delta=0.015, Conf=0.1, HistStd=0.015 -> Strength: {get_signal_strength(0.015, 0.1, 0.015)}")  # Moderate
    print(f"Delta=0.005, Conf=0.03, HistStd=0.025 -> Strength: {get_signal_strength(0.005, 0.03, 0.025)}")  # Weak

    print("\n--- Testing is_conflict ---")
    print(f"Model=0.01, HistAvg=-0.01 -> Conflict: {is_conflict(0.01, -0.01)}")  # True
    print(f"Model=0.01, HistAvg=0.008 -> Conflict: {is_conflict(0.01, 0.008)}")  # False
    print(f"Model=0.01, HistAvg=-0.001 -> Conflict: {is_conflict(0.01, -0.001)}")  # False (hist negligible)

    print("\n--- Testing get_confidence_hint ---")
    print(f"Score=0.25 -> Hint: {get_confidence_hint(0.25)}")
    print(f"Score=0.03 -> Hint: {get_confidence_hint(0.03)}")
    print(f"Score=0.01 -> Hint: {get_confidence_hint(0.01)}")

    print("\n--- Testing calculate_trade_levels ---")
    entry, direction, vol, rr = 100, 'long', 0.02, 2.0  # vol is 2% of entry for SL
    sl, tp = calculate_trade_levels(entry, direction, entry * vol, rr)  # Pass vol * entry as the SL distance basis
    print(f"Long: Entry={entry}, Vol (as amount)={entry * vol:.2f}, RR={rr} -> SL={sl}, TP={tp}")
    entry, direction, vol, rr = 100, 'short', 0.015, 1.5
    sl, tp = calculate_trade_levels(entry, direction, entry * vol, rr)
    print(f"Short: Entry={entry}, Vol (as amount)={entry * vol:.2f}, RR={rr} -> SL={sl}, TP={tp}")

    print("\n--- Testing similarity_analysis ---")
    # Dummy data for similarity_analysis
    hist_feat = pd.DataFrame({
        'f1': np.random.rand(50), 'f2': np.random.rand(50), 'f3': np.random.rand(50)
    })
    hist_deltas = pd.Series(np.random.randn(50) * 0.01)  # Random deltas around 0
    live_feat = pd.DataFrame({'f1': [0.5], 'f2': [0.6], 'f3': [0.4]})

    avg_d, std_d, hint = similarity_analysis(live_feat, hist_feat, hist_deltas, top_n_similar=5)
    print(f"Similarity: AvgDelta={avg_d:.4f}, StdDelta={std_d:.4f}, Hint='{hint}'")

    # Test with insufficient historical data
    avg_d, std_d, hint = similarity_analysis(live_feat, hist_feat.head(10), hist_deltas.head(10))
    print(f"Similarity (insufficient data): AvgDelta={avg_d}, StdDelta={std_d}, Hint='{hint}'")

    # Test with NaNs in live features
    live_feat_nan = pd.DataFrame({'f1': [0.5], 'f2': [np.nan], 'f3': [0.4]})
    avg_d, std_d, hint = similarity_analysis(live_feat_nan, hist_feat, hist_deltas)
    print(f"Similarity (NaN in live): AvgDelta={avg_d}, StdDelta={std_d}, Hint='{hint}'")
