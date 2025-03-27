import yfinance as yf
import pandas as pd
import requests
import datetime
import numpy as np
from scipy.stats import linregress
from tabulate import tabulate
import json
import os
import csv

# Initial settings
initial_portfolio_value = 100000  # Initial assets $100,000
TOLERANCE = 0.05  # 5% tolerance for weight difference

### Data collection functions ###
def get_current_vix():
    """Fetches the current VIX value."""
    try:
        vix = yf.Ticker("^VIX").history(period="1d")
        return vix['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return None

def get_fear_greed_index():
    """Fetches the CNN Fear & Greed Index."""
    current_date = datetime.date.today()
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{current_date.strftime('%Y-%m-%d')}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
    response = requests.get(url, headers=headers, timeout=10)
    data = response.json()
    fg_df = pd.DataFrame(data["fear_and_greed_historical"]["data"])
    fg_df['x'] = pd.to_datetime(fg_df['x'], unit='ms', errors='coerce').dt.date
    fg_df = fg_df.rename(columns={'x': 'Date', 'y': 'Fear & Greed Index'})
    fg_df['Fear & Greed Index'] = fg_df['Fear & Greed Index'].interpolate(method='linear')
    return fg_df['Fear & Greed Index'].iloc[-1]

def get_stock_data(ticker, period="max", interval="1d"):
    """Fetches stock data and calculates technical indicators."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.interpolate(method='linear')
        df['Close'] = df['Close'].round(2)
        df['ATR'] = calculate_atr(df, 14)
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'], 12, 26, 9)
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        df['SMA50'] = calculate_sma(df['Close'], 50)
        df['SMA200'] = calculate_sma(df['Close'], 200)
        df['Upper Band'], df['Middle Band'], df['Lower Band'] = calculate_bollinger_bands(df['Close'])
        df['Volume Change'] = df['Volume'].pct_change()
        df['Stochastic_K'], df['Stochastic_D'] = calculate_stochastic(df)
        df['OBV'] = calculate_obv(df)
        df['BB_width'] = (df['Upper Band'] - df['Lower Band']) / df['Middle Band']
        return df
    except Exception as e:
        print(f"Error loading {ticker} data: {e}")
        return None

def get_weekly_rsi(ticker, period="max"):
    """Calculates the weekly RSI."""
    try:
        weekly_df = yf.Ticker(ticker).history(period=period, interval="1wk")
        weekly_df.index = pd.to_datetime(weekly_df.index, errors='coerce')
        weekly_df = weekly_df.interpolate(method='linear')
        weekly_df['RSI'] = calculate_rsi(weekly_df['Close'], 14)
        return weekly_df['RSI'].iloc[-1]
    except Exception as e:
        print(f"Error calculating weekly RSI for {ticker}: {e}")
        return None

### Indicator calculation functions ###
def calculate_rsi(series, timeperiod=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.interpolate(method='linear')

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd.interpolate(method='linear'), macd_signal.interpolate(method='linear')

def calculate_sma(series, timeperiod=20):
    return series.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band.interpolate(method='linear'), sma.interpolate(method='linear'), lower_band.interpolate(method='linear')

def calculate_atr(df, timeperiod=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k.interpolate(method='linear'), d.interpolate(method='linear')

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index).interpolate(method='linear')

### Trend analysis functions ###
def get_rsi_trend(rsi_series, window=10):
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    return "Increasing" if slope > 0.1 else "Decreasing" if slope < -0.1 else "Stable"

def get_obv_trend(obv, obv_prev):
    return "Increasing" if obv > obv_prev else "Decreasing" if obv < obv_prev else "Stable"

def get_stochastic_trend(stochastic_k, stochastic_d):
    return "Above %D" if stochastic_k > stochastic_d else "Below %D" if stochastic_k < stochastic_d else "On %D"

def get_volume_change_trend(volume_change_series, window=5):
    if len(volume_change_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), volume_change_series[-window:])
    return "Increasing" if slope > 0.01 else "Decreasing" if slope < -0.01 else "Stable"

### Parameter management functions ###
def get_dynamic_default_params(vix):
    if vix is None or vix <= 30:
        return {
            "fg_buy": 25, "fg_sell": 75, "daily_rsi_buy": 30, "daily_rsi_sell": 70,
            "weekly_rsi_buy": 40, "weekly_rsi_sell": 60, "volume_change_strong_buy": 0.5,
            "volume_change_weak_buy": 0.2, "volume_change_sell": -0.2, "w_strong_buy": 2.0,
            "w_weak_buy": 1.0, "w_sell": 1.0, "stochastic_buy": 20, "stochastic_sell": 80,
            "obv_weight": 1.0, "bb_width_weight": 1.0
        }
    elif vix < 15:
        return {
            "fg_buy": 20, "fg_sell": 80, "daily_rsi_buy": 25, "daily_rsi_sell": 75,
            "weekly_rsi_buy": 35, "weekly_rsi_sell": 65, "volume_change_strong_buy": 0.4,
            "volume_change_weak_buy": 0.15, "volume_change_sell": -0.15, "w_strong_buy": 2.5,
            "w_weak_buy": 1.5, "w_sell": 1.5, "stochastic_buy": 25, "stochastic_sell": 75,
            "obv_weight": 1.2, "bb_width_weight": 1.2
        }
    else:
        return {
            "fg_buy": 30, "fg_sell": 70, "daily_rsi_buy": 35, "daily_rsi_sell": 65,
            "weekly_rsi_buy": 45, "weekly_rsi_sell": 55, "volume_change_strong_buy": 0.6,
            "volume_change_weak_buy": 0.25, "volume_change_sell": -0.25, "w_strong_buy": 1.5,
            "w_weak_buy": 0.8, "w_sell": 0.8, "stochastic_buy": 15, "stochastic_sell": 85,
            "obv_weight": 0.8, "bb_width_weight": 0.8
        }

def load_optimal_params(file_path="optimal_params.json", latest_version="2.0"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if "version" in data and "parameters" in data and data["version"] == latest_version:
                return data["parameters"]
            print("Parameters are outdated or format is incorrect. Using default values.")
    except Exception:
        print("Failed to load parameter file. Using dynamic default values.")
    return get_dynamic_default_params(get_current_vix())

### Portfolio management functions ###
def load_transactions(file_path="transactions.txt"):
    if not os.path.exists(file_path):
        print("transactions.txt file not found. Using initial assets.")
        return {}, 0
    try:
        df = pd.read_csv(file_path, sep='\s+', names=["date", "ticker", "action", "shares", "stock_price"])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        holdings = {}
        initial_investment = 0
        for i, row in df.iterrows():
            ticker, action, shares, price = row['ticker'], row['action'], row['shares'], row['stock_price']
            if i == 0 and action == "hold":
                initial_investment = shares * price
                holdings[ticker] = shares
            elif action == "buy":
                holdings[ticker] = holdings.get(ticker, 0) + shares
            elif action == "sell" and ticker in holdings and holdings[ticker] >= shares:
                holdings[ticker] -= shares
                if holdings[ticker] == 0:
                    del holdings[ticker]
        return {k: v for k, v in holdings.items() if v > 0}, initial_investment
    except Exception as e:
        print(f"Error loading transaction history: {e}")
        return {}, 0

def calculate_portfolio_metrics(current_holdings, tsla_close, tsll_close, initial_investment):
    tsla_shares = current_holdings.get("TSLA", 0)
    tsll_shares = current_holdings.get("TSLL", 0)
    tsla_value = tsla_shares * tsla_close
    tsll_value = tsll_shares * tsll_close
    total_value = tsla_value + tsll_value
    tsla_weight = tsla_value / total_value if total_value > 0 else 0
    tsll_weight = tsll_value / total_value if total_value > 0 else 0
    returns = ((total_value - initial_investment) / initial_investment * 100) if initial_investment > 0 else 0
    return total_value, tsla_value, tsll_value, tsla_weight, tsll_weight, returns

def load_previous_recommendation(file_path="portfolio_log.csv"):
    """Loads the most recent recommended TSLL weight and reasons from portfolio_log.csv."""
    if not os.path.exists(file_path):
        return None, None, None
    try:
        df = pd.read_csv(file_path, names=["date", "tsll_weight", "reasons"])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        last_entry = df.iloc[-1]
        prev_date = last_entry['date'].strftime('%Y-%m-%d')
        tsll_weight = float(last_entry['tsll_weight'])
        reasons_str = last_entry['reasons']
        # Parse the reasons string into a structured list
        reasons = []
        current_section = None
        for item in reasons_str.split("; "):
            item = item.strip()
            if item.startswith("Buy Signals"):
                current_section = "Buy Signals (Potential increase in TSLL weight):"
                reasons.append(current_section)
            elif item.startswith("Sell Signals"):
                current_section = "Sell Signals (Potential decrease in TSLL weight):"
                reasons.append(current_section)
            elif item.startswith("- ") and current_section:
                reasons.append(f"  - {item[2:]}")
        return prev_date, tsll_weight, reasons
    except Exception as e:
        print(f"Error loading portfolio log: {e}")
        return None, None, None

def save_recommendation(date, tsll_weight, reasons, file_path="portfolio_log.csv"):
    """Saves the recommended TSLL weight to portfolio_log.csv."""
    with open(file_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date, tsll_weight, "; ".join(reasons)])

def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, macd_histogram, volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, current_tsll_weight, optimal_params, data_date):
    """Calculates the target weight for TSLL and logs it."""
    base_weight = current_tsll_weight
    atr_normalized = atr / close if close > 0 else 0
    volume_change_strong_buy = optimal_params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = optimal_params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = optimal_params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = {
        f"Fear & Greed Index ≤ {optimal_params['fg_buy']}": fear_greed <= optimal_params["fg_buy"],
        f"Daily RSI < {optimal_params['daily_rsi_buy']}": daily_rsi < optimal_params["daily_rsi_buy"],
        f"Weekly RSI < {optimal_params['weekly_rsi_buy']}": weekly_rsi < optimal_params["weekly_rsi_buy"],
        "MACD > Signal (Signal < 0)": (macd > macd_signal) and (macd_signal < 0),
        "MACD Histogram > 0": macd_histogram > 0,
        f"Volume Change > {volume_change_strong_buy:.2f} (Strong Buy)": volume_change > volume_change_strong_buy,
        f"Volume Change > {volume_change_weak_buy:.2f} (Weak Buy)": volume_change > volume_change_weak_buy,
        "Close < Lower Band": close < lower_band,
        "RSI Increasing & Close > SMA200": (daily_rsi_trend == "Increasing") and (close > sma200),
        f"Stochastic %K < {optimal_params['stochastic_buy']}": stochastic_k < optimal_params["stochastic_buy"],
        "OBV Increasing": obv > obv_prev,
        "BB Width < 0.05": bb_width < 0.05
    }

    sell_conditions = {
        f"Fear & Greed Index ≥ {optimal_params['fg_sell']}": fear_greed >= optimal_params["fg_sell"],
        f"Daily RSI > {optimal_params['daily_rsi_sell']}": daily_rsi > optimal_params["daily_rsi_sell"],
        f"Weekly RSI > {optimal_params['weekly_rsi_sell']}": weekly_rsi > optimal_params["weekly_rsi_sell"],
        "MACD < Signal (Signal > 0)": (macd < macd_signal) and (macd_signal > 0),
        "MACD Histogram < 0": macd_histogram < 0,
        f"Volume Change < {volume_change_sell:.2f}": volume_change < volume_change_sell,
        "Close > Upper Band": close > upper_band,
        "RSI Decreasing & Close < SMA200": (daily_rsi_trend == "Decreasing") and (close < sma200),
        f"Stochastic %K > {optimal_params['stochastic_sell']}": stochastic_k > optimal_params["stochastic_sell"],
        "OBV Decreasing": obv < obv_prev,
        "BB Width > 0.15": bb_width > 0.15
    }

    buy_reasons = [cond for cond, val in buy_conditions.items() if val]
    sell_reasons = [cond for cond, val in sell_conditions.items() if val]

    w_strong_buy = optimal_params["w_strong_buy"]
    w_weak_buy = optimal_params["w_weak_buy"]
    w_sell = optimal_params["w_sell"]
    obv_weight = optimal_params["obv_weight"]
    bb_width_weight = optimal_params["bb_width_weight"]

    strong_buy_count = sum(1 for r in buy_reasons if "Strong Buy" in r)
    weak_buy_count = sum(1 for r in buy_reasons if "Weak Buy" in r and "Strong Buy" not in r)
    other_buy_count = len(buy_reasons) - strong_buy_count - weak_buy_count
    sell_count = len(sell_reasons)

    buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * (weak_buy_count + other_buy_count) + obv_weight * ("OBV Increasing" in buy_reasons) + bb_width_weight * ("BB Width < 0.05" in buy_reasons)) * 0.1
    sell_adjustment = (w_sell * sell_count + obv_weight * ("OBV Decreasing" in sell_reasons) + bb_width_weight * ("BB Width > 0.15" in sell_reasons)) * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    reasons = []
    if buy_reasons:
        reasons.append("Buy Signals (Potential increase in TSLL weight):")
        reasons.extend(f"  - {r}" for r in buy_reasons)
    if sell_reasons:
        reasons.append("Sell Signals (Potential decrease in TSLL weight):")
        reasons.extend(f"  - {r}" for r in sell_reasons)
    if not reasons:
        reasons.append("- No significant signals detected.")

    # Save the recommendation
    save_recommendation(data_date, target_weight, reasons)

    return target_weight, reasons

def adjust_portfolio(target_tsla_weight, target_tsll_weight, current_holdings, total_value, tsla_close, tsll_close):
    current_tsla_shares = current_holdings.get("TSLA", 0)
    current_tsll_shares = current_holdings.get("TSLL", 0)
    target_tsla_shares = int((target_tsla_weight * total_value) / tsla_close)
    target_tsll_shares = int((target_tsll_weight * total_value) / tsll_close)

    tsla_diff = target_tsla_shares - current_tsla_shares
    tsll_diff = target_tsll_shares - current_tsll_shares

    if tsla_diff == 0 and tsll_diff == 0:
        print(" - No adjustment needed")
    else:
        print(f" - TSLA: {'Buy' if tsla_diff > 0 else 'Sell'} {abs(tsla_diff)} shares (target weight {target_tsla_weight*100:.2f}%)" if tsla_diff != 0 else " - TSLA: No adjustment needed")
        print(f" - TSLL: {'Buy' if tsll_diff > 0 else 'Sell'} {abs(tsll_diff)} shares (target weight {target_tsll_weight*100:.2f}%)" if tsll_diff != 0 else " - TSLL: No adjustment needed")

### Main execution function ###
def analyze_and_recommend():
    print("Loading data...")
    optimal_params = load_optimal_params()
    fear_greed = get_fear_greed_index()
    tsla_df = get_stock_data("TSLA")
    tsll_df = get_stock_data("TSLL")
    weekly_rsi = get_weekly_rsi("TSLA")

    if tsla_df is None or tsll_df is None or weekly_rsi is None:
        print("Data loading failed. Exiting.")
        return

    tsla_close = tsla_df['Close'].iloc[-1]
    tsll_close = tsll_df['Close'].iloc[-1]
    data_date = tsla_df.index[-1].strftime('%Y-%m-%d')

    # Calculate indicators
    daily_rsi = tsla_df['RSI'].iloc[-1]
    daily_rsi_trend = get_rsi_trend(tsla_df['RSI'].tail(10))
    sma50, sma200 = tsla_df['SMA50'].iloc[-1], tsla_df['SMA200'].iloc[-1]
    macd, macd_signal = tsla_df['MACD'].iloc[-1], tsla_df['MACD_signal'].iloc[-1]
    macd_histogram = tsla_df['MACD_histogram'].iloc[-1]
    upper_band, lower_band = tsla_df['Upper Band'].iloc[-1], tsla_df['Lower Band'].iloc[-1]
    volume_change = tsla_df['Volume Change'].iloc[-1]
    atr = tsla_df['ATR'].iloc[-1]
    stochastic_k, stochastic_d = tsla_df['Stochastic_K'].iloc[-1], tsla_df['Stochastic_D'].iloc[-1]
    obv = tsla_df['OBV'].iloc[-1]
    obv_prev = tsla_df['OBV'].iloc[-2] if len(tsla_df) > 1 else obv
    bb_width = tsla_df['BB_width'].iloc[-1]

    # Calculate trends and notes
    fear_greed_note = "Low" if fear_greed < 30 else "High" if fear_greed > 70 else "Neutral"
    weekly_rsi_note = "Oversold" if weekly_rsi < 30 else "Overbought" if weekly_rsi > 70 else "Neutral"
    close_note = []
    if tsla_close > sma50:
        close_note.append("Above SMA50")
    else:
        close_note.append("Below SMA50")
    if tsla_close > sma200:
        close_note.append("Above SMA200")
    else:
        close_note.append("Below SMA200")
    close_note = ", ".join(close_note)
    volume_change_trend = get_volume_change_trend(tsla_df['Volume Change'].tail(5))
    stochastic_trend = get_stochastic_trend(stochastic_k, stochastic_d)
    obv_trend = get_obv_trend(obv, obv_prev)
    bb_width_note = "Low" if bb_width < 0.05 else "High" if bb_width > 0.15 else "Medium"

    # Indicators table
    indicators = [
        ["Fear & Greed Index", f"{fear_greed:.2f}", fear_greed_note],
        ["Daily RSI", f"{daily_rsi:.2f}", daily_rsi_trend],
        ["Weekly RSI", f"{weekly_rsi:.2f}", weekly_rsi_note],
        ["TSLA Close", f"${tsla_close:.2f}", close_note],
        ["SMA50", f"${sma50:.2f}", "N/A"],
        ["SMA200", f"${sma200:.2f}", "N/A"],
        ["Upper Bollinger Band", f"${upper_band:.2f}", "N/A"],
        ["Lower Bollinger Band", f"${lower_band:.2f}", "N/A"],
        ["Volume Change", f"{volume_change:.2%}", volume_change_trend],
        ["ATR", f"${atr:.2f}", "N/A"],
        ["Stochastic %K", f"{stochastic_k:.2f}", stochastic_trend],
        ["Stochastic %D", f"{stochastic_d:.2f}", "N/A"],
        ["OBV", f"{obv:.0f}", obv_trend],
        ["BB Width", f"{bb_width:.4f}", bb_width_note],
        ["MACD Histogram", f"{macd_histogram:.2f}", "N/A"]
    ]
    print(f"\n### Market Indicators (as of {data_date})")
    print(tabulate(indicators, headers=["Indicator", "Value", "Trend/Notes"], tablefmt="fancy_grid"))

    # Stock prices display
    tsla_prev_close = tsla_df['Close'].iloc[-2] if len(tsla_df) >= 2 else tsla_close
    tsla_change = (tsla_close - tsla_prev_close) / tsla_prev_close * 100 if len(tsla_df) >= 2 else 0
    tsll_prev_close = tsll_df['Close'].iloc[-2] if len(tsll_df) >= 2 else tsll_close
    tsll_change = (tsll_close - tsll_prev_close) / tsll_prev_close * 100 if len(tsll_df) >= 2 else 0

    print("\n### Current Stock Prices")
    print(f"- **TSLA Close**: ${tsla_close:.2f} (Change: {tsla_change:.2f}%)")
    print(f"- **TSLL Close**: ${tsll_close:.2f} (Change: {tsll_change:.2f}%)")

    # Portfolio display
    current_holdings, initial_investment = load_transactions()
    total_value, tsla_value, tsll_value, tsla_weight, tsll_weight, returns = calculate_portfolio_metrics(current_holdings, tsla_close, tsll_close, initial_investment)
    profit_loss = total_value - initial_investment
    print("\n### Current Portfolio")
    print(f"- Initial Investment: ${initial_investment:.2f}")
    print(f"- Current Total Value: ${total_value:.2f}")
    print(f"- Profit/Loss: ${profit_loss:.2f} ({returns:.2f}%)")
    print(f"- TSLA: {current_holdings.get('TSLA', 0)} shares, value: ${tsla_value:.2f} ({tsla_weight*100:.2f}%)")
    print(f"- TSLL: {current_holdings.get('TSLL', 0)} shares, value: ${tsll_value:.2f} ({tsll_weight*100:.2f}%)")

    # Load previous recommendation
    prev_date, prev_tsll_weight, prev_reasons = load_previous_recommendation()
    prev_tsla_weight = 1 - prev_tsll_weight if prev_tsll_weight is not None else None

    # Compare with previous recommendation
    if prev_date and prev_tsll_weight is not None:
        print(f"\n### Previous Recommendation (as of {prev_date})")
        print(f"- TSLA: {prev_tsla_weight*100:.2f}%")
        print(f"- TSLL: {prev_tsll_weight*100:.2f}%")

        tsla_weight_diff = abs(tsla_weight - prev_tsla_weight)
        tsll_weight_diff = abs(tsll_weight - prev_tsll_weight)
        print(f"\n### Difference from Previous Recommendation")
        print(f"- TSLA Weight Difference: {tsla_weight_diff*100:.2f}%")
        print(f"- TSLL Weight Difference: {tsll_weight_diff*100:.2f}%")

        # Check if the previous recommendation is from today
        if prev_date == data_date:
            if tsla_weight_diff <= TOLERANCE and tsll_weight_diff <= TOLERANCE:
                print("\n### Portfolio Adjustment Suggestion")
                print(" - Portfolio is within tolerance (5%) of today's recommendation. No adjustment needed.")
            else:
                print("\n### Portfolio Adjustment Suggestion")
                print(f" - Portfolio deviates from today's recommendation by more than {TOLERANCE*100}%. Adjusting to today's weights:")
                adjust_portfolio(prev_tsla_weight, prev_tsll_weight, current_holdings, total_value, tsla_close, tsll_close)
                print("\n### Adjustment Reasons")
                print(f" - Reverting to previous recommendation from {prev_date}:")
                for reason in prev_reasons:
                    print(reason)
        else:
            # If previous recommendation is from a different day, provide a new recommendation
            print("\n### New Recommendation Based on Latest Market Data")
            target_tsll_weight, reasons = get_target_tsll_weight(
                fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, tsla_close, sma50, sma200, macd, macd_signal, macd_histogram,
                volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width,
                tsll_weight, optimal_params, data_date
            )
            target_tsla_weight = 1 - target_tsll_weight
            print(f"- TSLA: {target_tsla_weight*100:.2f}%")
            print(f"- TSLL: {target_tsll_weight*100:.2f}%")
            print("\n### Portfolio Adjustment Suggestion")
            adjust_portfolio(target_tsla_weight, target_tsll_weight, current_holdings, total_value, tsla_close, tsll_close)
            print("\n### Adjustment Reasons")
            for reason in reasons:
                print(reason)
    else:
        # No previous recommendation exists
        print("\n### Recommended Portfolio Weights")
        target_tsll_weight, reasons = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, tsla_close, sma50, sma200, macd, macd_signal, macd_histogram,
            volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width,
            tsll_weight, optimal_params, data_date
        )
        target_tsla_weight = 1 - target_tsll_weight
        print(f"- TSLA: {target_tsla_weight*100:.2f}%")
        print(f"- TSLL: {target_tsll_weight*100:.2f}%")
        print("\n### Portfolio Adjustment Suggestion")
        adjust_portfolio(target_tsla_weight, target_tsll_weight, current_holdings, total_value, tsla_close, tsll_close)
        print("\n### Adjustment Reasons")
        for reason in reasons:
            print(reason)

if __name__ == "__main__":
    analyze_and_recommend()
