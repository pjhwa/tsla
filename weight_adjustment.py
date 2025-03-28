import json

def load_params(file_path="optimal_params.json"):
    """설정파일에서 파라미터를 로드"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data["parameters"]
    except Exception as e:
        raise Exception(f"Failed to load parameters from {file_path}: {e}")

def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma5, sma10, sma50, sma200, macd, macd_signal, macd_histogram, volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, rsi5, macd_short, macd_signal_short, vwap, current_tsll_weight, params, data_date):
    """TSLL 목표 비중 계산 및 이유 생성"""
    base_weight = current_tsll_weight
    atr_normalized = atr / close if close > 0 else 0
    volume_change_strong_buy = params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = {
        f"Fear & Greed Index ≤ {params['fg_buy']}": fear_greed <= params["fg_buy"],
        f"Daily RSI < {params['daily_rsi_buy']}": daily_rsi < params["daily_rsi_buy"],
        f"Weekly RSI < {params['weekly_rsi_buy']}": weekly_rsi < params["weekly_rsi_buy"],
        "MACD > Signal (Signal < 0)": (macd > macd_signal) and (macd_signal < 0),
        "MACD Histogram > 0": macd_histogram > 0,
        f"Volume Change > {volume_change_strong_buy:.2f} (Strong Buy)": volume_change > volume_change_strong_buy,
        f"Volume Change > {volume_change_weak_buy:.2f} (Weak Buy)": volume_change > volume_change_weak_buy,
        "Close < Lower Band": close < lower_band,
        "RSI Increasing & Close > SMA200": (daily_rsi_trend == "Increasing") and (close > sma200),
        f"Stochastic %K < {params['stochastic_buy']}": stochastic_k < params["stochastic_buy"],
        "OBV Increasing": obv > obv_prev,
        f"BB Width < {params['bb_width_low']}": bb_width < params["bb_width_low"],
        "SMA5 > SMA10": sma5 > sma10,
        f"Short RSI < {params['short_rsi_buy']}": rsi5 < params["short_rsi_buy"],
        "Short MACD > Signal": macd_short > macd_signal_short,
        "Close > VWAP": close > vwap
    }

    sell_conditions = {
        f"Fear & Greed Index ≥ {params['fg_sell']}": fear_greed >= params["fg_sell"],
        f"Daily RSI > {params['daily_rsi_sell']}": daily_rsi > params["daily_rsi_sell"],
        f"Weekly RSI > {params['weekly_rsi_sell']}": weekly_rsi > params["weekly_rsi_sell"],
        "MACD < Signal (Signal > 0)": (macd < macd_signal) and (macd_signal > 0),
        "MACD Histogram < 0": macd_histogram < 0,
        f"Volume Change < {volume_change_sell:.2f}": volume_change < volume_change_sell,
        "Close > Upper Band": close > upper_band,
        "RSI Decreasing & Close < SMA200": (daily_rsi_trend == "Decreasing") and (close < sma200),
        f"Stochastic %K > {params['stochastic_sell']}": stochastic_k > params["stochastic_sell"],
        "OBV Decreasing": obv < obv_prev,
        f"BB Width > {params['bb_width_high']}": bb_width > params["bb_width_high"],
        "SMA5 < SMA10": sma5 < sma10,
        f"Short RSI > {params['short_rsi_sell']}": rsi5 > params["short_rsi_sell"],
        "Short MACD < Signal": macd_short < macd_signal_short,
        "Close < VWAP": close < vwap
    }

    buy_reasons = [cond for cond, val in buy_conditions.items() if val]
    sell_reasons = [cond for cond, val in sell_conditions.items() if val]

    w_strong_buy = params["w_strong_buy"]
    w_weak_buy = params["w_weak_buy"]
    w_sell = params["w_sell"]
    obv_weight = params["obv_weight"]
    bb_width_weight = params["bb_width_weight"]
    w_short_buy = params["w_short_buy"]
    w_short_sell = params["w_short_sell"]

    strong_buy_count = sum(1 for r in buy_reasons if "Strong Buy" in r)
    weak_buy_count = sum(1 for r in buy_reasons if "Weak Buy" in r and "Strong Buy" not in r)
    other_buy_count = len(buy_reasons) - strong_buy_count - weak_buy_count
    sell_count = len(sell_reasons)
    short_buy_count = sum(1 for r in buy_reasons if "Short RSI" in r or "Short MACD" in r or "SMA5 > SMA10" in r or "Close > VWAP" in r)
    short_sell_count = sum(1 for r in sell_reasons if "Short RSI" in r or "Short MACD" in r or "SMA5 < SMA10" in r or "Close < VWAP" in r)

    buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * weak_buy_count + w_weak_buy * other_buy_count +
                      obv_weight * ("OBV Increasing" in buy_reasons) + bb_width_weight * (f"BB Width < {params['bb_width_low']}" in buy_reasons) +
                      w_short_buy * short_buy_count) * 0.1
    sell_adjustment = (w_sell * sell_count + obv_weight * ("OBV Decreasing" in sell_reasons) +
                       bb_width_weight * (f"BB Width > {params['bb_width_high']}" in sell_reasons) + w_short_sell * short_sell_count) * 0.1

    reasons_list = []

    if rsi5 > params["short_rsi_sell"]:
        buy_adjustment *= 0.5
        sell_reasons.append("Overbought RSI5 detected")
    elif rsi5 < params["short_rsi_buy"]:
        buy_adjustment *= 1.5
        buy_reasons.append("Oversold RSI5 detected")

    atr_percentage = atr / close if close > 0 else 0
    volatility_factor = 1.0
    if atr_percentage > 0.05:
        volatility_factor = 0.7
        sell_reasons.append("High volatility detected (ATR > 5%)")
    elif atr_percentage < 0.02:
        volatility_factor = 1.2
        buy_reasons.append("Low volatility detected (ATR < 2%)")

    preliminary_target_weight = base_weight + buy_adjustment - sell_adjustment
    preliminary_target_weight *= volatility_factor

    weight_change = preliminary_target_weight - base_weight
    MAX_WEIGHT_CHANGE = 0.2  # 일일 최대 비중 변동폭 20%
    if abs(weight_change) > MAX_WEIGHT_CHANGE:
        target_weight = base_weight + (MAX_WEIGHT_CHANGE if weight_change > 0 else -MAX_WEIGHT_CHANGE)
        reasons_list.append(f"Weight change limited to {MAX_WEIGHT_CHANGE*100:.0f}% per day")
    else:
        target_weight = preliminary_target_weight

    target_weight = max(0.0, min(target_weight, 1.0))

    if buy_reasons:
        reasons_list.append("Buy Signals (Potential increase in TSLL weight):")
        reasons_list.extend(f"  - {r}" for r in buy_reasons)
    if sell_reasons:
        reasons_list.append("Sell Signals (Potential decrease in TSLL weight):")
        reasons_list.extend(f"  - {r}" for r in sell_reasons)
    if not buy_reasons and not sell_reasons:
        reasons_list.append("- No significant signals detected.")

    return target_weight, reasons_list
