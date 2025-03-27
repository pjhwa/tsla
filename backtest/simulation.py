import pandas as pd
import numpy as np
import json
from scipy.stats import linregress
import os
from datetime import datetime, timedelta
import argparse
from tabulate import tabulate
import logging

# 상수 정의
INITIAL_VALUE = 100000  # 초기 자산 $100,000
TRANSACTION_COST = 0.001  # 거래 비용 0.1%

def setup_logging(start_date, end_date):
    """로그 설정 및 파일명 반환"""
    log_filename = f"simulation-{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return log_filename

def load_and_preprocess_data(file_path, date_column='Date', volume_column='Volume'):
    """CSV 파일을 로드하고 전처리"""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if volume_column in df.columns:
        df[volume_column] = df[volume_column].str.replace(',', '').astype(float)
    return df

def merge_dataframes(tsla_df, tsll_df, fear_greed_df, start_date, end_date):
    """데이터프레임 병합 및 필터링"""
    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.interpolate(method='linear')
    return data[(data.index >= start_date) & (data.index <= end_date)]

def calculate_rsi(series, timeperiod=14):
    """RSI 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).interpolate(method='linear')

def calculate_sma(series, timeperiod=20):
    """SMA 계산"""
    return series.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD 계산"""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd.interpolate(method='linear'), macd_signal.interpolate(method='linear'), macd_histogram.interpolate(method='linear')

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """볼린저 밴드 계산"""
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band.interpolate(method='linear'), sma.interpolate(method='linear'), lower_band.interpolate(method='linear')

def calculate_atr(df, timeperiod=14):
    """ATR 계산"""
    high_low = df['High_TSLA'] - df['Low_TSLA']
    high_close = np.abs(df['High_TSLA'] - df['Close_TSLA'].shift())
    low_close = np.abs(df['Low_TSLA'] - df['Close_TSLA'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_stochastic(df, k_period=14, d_period=3):
    """스토캐스틱 계산"""
    low_min = df['Low_TSLA'].rolling(window=k_period).min()
    high_max = df['High_TSLA'].rolling(window=k_period).max()
    k = 100 * ((df['Close_TSLA'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k.interpolate(method='linear'), d.interpolate(method='linear')

def calculate_obv(close, volume):
    """OBV 계산"""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index).interpolate(method='linear')

def calculate_vwap(df):
    """VWAP 계산"""
    vwap = (df['Close_TSLA'] * df['Volume_TSLA']).cumsum() / df['Volume_TSLA'].cumsum()
    return vwap

def get_rsi_trend(rsi_series, window=10):
    """RSI 추세 분석"""
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    return "Increasing" if slope > 0.1 else "Decreasing" if slope < -0.1 else "Stable"

def load_params(file_path="optimal_params.json"):
    """파라미터 로드"""
    latest_version = "2.0"
    default_params = {
        "fg_buy": 25, "fg_sell": 75, "daily_rsi_buy": 30, "daily_rsi_sell": 70,
        "weekly_rsi_buy": 40, "weekly_rsi_sell": 60, "volume_change_strong_buy": 0.5,
        "volume_change_weak_buy": 0.2, "volume_change_sell": -0.2, "w_strong_buy": 2.0,
        "w_weak_buy": 1.0, "w_sell": 1.0, "stochastic_buy": 20, "stochastic_sell": 80,
        "obv_weight": 1.0, "bb_width_weight": 1.0, "short_rsi_buy": 25, "short_rsi_sell": 75,
        "bb_width_low": 0.1, "bb_width_high": 0.2, "w_short_buy": 1.5, "w_short_sell": 1.5
    }
    if not os.path.exists(file_path):
        logging.warning(f"{file_path} file not found. Using default parameters.")
        return default_params
    try:
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
        if "version" in loaded_data and loaded_data["version"] == latest_version:
            return loaded_data["parameters"]
        else:
            logging.warning(f"Version mismatch in {file_path}. Using default parameters.")
            return default_params
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error loading {file_path}: {e}. Using default parameters.")
        return default_params

def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma5, sma10, sma50, sma200, macd, macd_signal, macd_histogram, volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, rsi5, macd_short, macd_signal_short, vwap, current_tsll_weight, params):
    """TSLL 목표 비중 계산 및 조정 사유 반환"""
    base_weight = current_tsll_weight
    atr_normalized = atr / close if close > 0 else 0
    volume_change_strong_buy = params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = [
        ("Fear & Greed <= fg_buy", fear_greed <= params["fg_buy"]),
        ("Daily RSI < daily_rsi_buy", daily_rsi < params["daily_rsi_buy"]),
        ("Weekly RSI < weekly_rsi_buy", weekly_rsi < params["weekly_rsi_buy"]),
        ("MACD > Signal and Signal < 0", (macd > macd_signal) and (macd_signal < 0)),
        ("MACD Histogram > 0", macd_histogram > 0),
        ("Volume Change > Strong Buy", volume_change > volume_change_strong_buy),
        ("Volume Change > Weak Buy", volume_change > volume_change_weak_buy),
        ("Close < Lower Band", close < lower_band),
        ("RSI Increasing and Close > SMA200", (daily_rsi_trend == "Increasing") and (close > sma200)),
        ("Stochastic K < stochastic_buy", stochastic_k < params["stochastic_buy"]),
        ("OBV > OBV_prev", obv > obv_prev),
        ("BB Width < bb_width_low", bb_width < params["bb_width_low"]),
        ("SMA5 > SMA10", sma5 > sma10),
        ("RSI5 < short_rsi_buy", rsi5 < params["short_rsi_buy"]),
        ("MACD Short > Signal Short", macd_short > macd_signal_short),
        ("Close > VWAP", close > vwap)
    ]

    sell_conditions = [
        ("Fear & Greed >= fg_sell", fear_greed >= params["fg_sell"]),
        ("Daily RSI > daily_rsi_sell", daily_rsi > params["daily_rsi_sell"]),
        ("Weekly RSI > weekly_rsi_sell", weekly_rsi > params["weekly_rsi_sell"]),
        ("MACD < Signal and Signal > 0", (macd < macd_signal) and (macd_signal > 0)),
        ("MACD Histogram < 0", macd_histogram < 0),
        ("Volume Change < Sell", volume_change < volume_change_sell),
        ("Close > Upper Band", close > upper_band),
        ("RSI Decreasing and Close < SMA200", (daily_rsi_trend == "Decreasing") and (close < sma200)),
        ("Stochastic K > stochastic_sell", stochastic_k > params["stochastic_sell"]),
        ("OBV < OBV_prev", obv < obv_prev),
        ("BB Width > bb_width_high", bb_width > params["bb_width_high"]),
        ("SMA5 < SMA10", sma5 < sma10),
        ("RSI5 > short_rsi_sell", rsi5 > params["short_rsi_sell"]),
        ("MACD Short < Signal Short", macd_short < macd_signal_short),
        ("Close < VWAP", close < vwap)
    ]

    w_strong_buy = params["w_strong_buy"]
    w_weak_buy = params["w_weak_buy"]
    w_sell = params["w_sell"]
    obv_weight = params["obv_weight"]
    bb_width_weight = params["bb_width_weight"]
    w_short_buy = params["w_short_buy"]
    w_short_sell = params["w_short_sell"]

    short_buy_count = sum(cond[1] for cond in buy_conditions[11:])
    short_sell_count = sum(cond[1] for cond in sell_conditions[11:])

    buy_adjustment = (w_strong_buy * buy_conditions[5][1] + w_weak_buy * (buy_conditions[6][1] + sum(cond[1] for cond in buy_conditions[:5]) + sum(cond[1] for cond in buy_conditions[7:11])) + obv_weight * buy_conditions[10][1] + bb_width_weight * buy_conditions[11][1] + w_short_buy * short_buy_count) * 0.1
    sell_adjustment = (w_sell * sum(cond[1] for cond in sell_conditions[:6]) + obv_weight * sell_conditions[9][1] + bb_width_weight * sell_conditions[10][1] + w_short_sell * short_sell_count) * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    reasons = [cond[0] for cond in buy_conditions if cond[1]] + [cond[0] for cond in sell_conditions if cond[1]]
    return target_weight, reasons

def adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row):
    """포트폴리오 조정 및 거래 내역 반환"""
    total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
    required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])
    required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])

    trades = []

    # 먼저 TSLL 조정: 매도하여 현금 확보
    if required_tsll_shares < holdings['TSLL']:
        sell_shares = holdings['TSLL'] - required_tsll_shares
        proceeds = sell_shares * row['Close_TSLL'] * (1 - TRANSACTION_COST)
        cash += proceeds
        holdings['TSLL'] -= sell_shares
        trades.append(f"Sell {sell_shares} TSLL shares at ${row['Close_TSLL']:.2f} (Proceeds: ${proceeds:.2f})")
    elif required_tsll_shares > holdings['TSLL']:
        buy_shares = required_tsll_shares - holdings['TSLL']
        cost = buy_shares * row['Close_TSLL'] * (1 + TRANSACTION_COST)
        if cash >= cost:
            cash -= cost
            holdings['TSLL'] += buy_shares
            trades.append(f"Buy {buy_shares} TSLL shares at ${row['Close_TSLL']:.2f} (Cost: ${cost:.2f})")
        else:
            buy_shares = int(cash / (row['Close_TSLL'] * (1 + TRANSACTION_COST)))
            cost = buy_shares * row['Close_TSLL'] * (1 + TRANSACTION_COST)
            cash -= cost
            holdings['TSLL'] += buy_shares
            trades.append(f"Buy {buy_shares} TSLL shares at ${row['Close_TSLL']:.2f} (Cost: ${cost:.2f})")

    # TSLA 조정: 남은 현금으로 최대한 매수
    if required_tsla_shares > holdings['TSLA']:
        buy_shares = required_tsla_shares - holdings['TSLA']
        cost = buy_shares * row['Close_TSLA'] * (1 + TRANSACTION_COST)
        if cash >= cost:
            cash -= cost
            holdings['TSLA'] += buy_shares
            trades.append(f"Buy {buy_shares} TSLA shares at ${row['Close_TSLA']:.2f} (Cost: ${cost:.2f})")
        else:
            buy_shares = int(cash / (row['Close_TSLA'] * (1 + TRANSACTION_COST)))
            cost = buy_shares * row['Close_TSLA'] * (1 + TRANSACTION_COST)
            cash -= cost
            holdings['TSLA'] += buy_shares
            trades.append(f"Buy {buy_shares} TSLA shares at ${row['Close_TSLA']:.2f} (Cost: ${cost:.2f})")
    elif required_tsla_shares < holdings['TSLA']:
        sell_shares = holdings['TSLA'] - required_tsla_shares
        proceeds = sell_shares * row['Close_TSLA'] * (1 - TRANSACTION_COST)
        cash += proceeds
        holdings['TSLA'] -= sell_shares
        trades.append(f"Sell {sell_shares} TSLA shares at ${row['Close_TSLA']:.2f} (Proceeds: ${proceeds:.2f})")

    return holdings, cash, trades

def load_data(start_date, end_date):
    """데이터 로드 및 지표 계산"""
    tsla_df = load_and_preprocess_data('TSLA-history-2y.csv')
    tsll_df = load_and_preprocess_data('TSLL-history-2y.csv')
    fear_greed_df = load_and_preprocess_data('fear_greed_2years.csv', date_column='date', volume_column=None)
    data = merge_dataframes(tsla_df, tsll_df, fear_greed_df, start_date, end_date)

    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA5_TSLA'] = calculate_sma(data['Close_TSLA'], 5)
    data['SMA10_TSLA'] = calculate_sma(data['Close_TSLA'], 10)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data['Close_TSLA'])
    data['Upper Band_TSLA'] = upper_band
    data['Middle Band_TSLA'] = middle_band
    data['Lower Band_TSLA'] = lower_band
    data['BB_width_TSLA'] = (upper_band - lower_band) / middle_band
    data['MACD_TSLA'], data['MACD_signal_TSLA'], data['MACD_histogram_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['MACD_short_TSLA'], data['MACD_signal_short_TSLA'], _ = calculate_macd(data['Close_TSLA'], 5, 35, 5)
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']], 14)
    data['Weekly RSI_TSLA'] = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14).reindex(data.index, method='ffill').fillna(50)
    data['Stochastic_K_TSLA'], data['Stochastic_D_TSLA'] = calculate_stochastic(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']])
    data['OBV_TSLA'] = calculate_obv(data['Close_TSLA'], data['Volume_TSLA'])
    data['VWAP_TSLA'] = calculate_vwap(data)
    data['RSI5_TSLA'] = calculate_rsi(data['Close_TSLA'], 5)

    return data

def simulate_portfolio(start_date, end_date, params):
    """포트폴리오 시뮬레이션"""
    data = load_data(start_date, end_date)
    if data.empty:
        logging.error("No data available for the specified period.")
        print("No data available for the specified period.")
        return None, None, None, None, None, None, None

    log_filename = setup_logging(start_date, end_date)
    logging.info(f"Simulation started: {start_date.date()} to {end_date.date()}")
    logging.info(f"Initial Portfolio Value: ${INITIAL_VALUE:.2f}")

    cash = INITIAL_VALUE
    holdings = {'TSLA': 0, 'TSLL': 0}
    current_tsll_weight = 0.0
    prev_target_tsll_weight = 0.0
    portfolio_values = []
    dates = []

    for i in range(len(data)):
        row = data.iloc[i]
        date = row.name.date()

        # 당일 데이터를 기반으로 목표 비중 계산 (After Market)
        fear_greed = row['y']
        daily_rsi = row['RSI_TSLA']
        weekly_rsi = row['Weekly RSI_TSLA']
        daily_rsi_trend = get_rsi_trend(data['RSI_TSLA'].iloc[max(0, i-10):i+1])
        close = row['Close_TSLA']
        sma5 = row['SMA5_TSLA']
        sma10 = row['SMA10_TSLA']
        sma50 = row['SMA50_TSLA']
        sma200 = row['SMA200_TSLA']
        macd = row['MACD_TSLA']
        macd_signal = row['MACD_signal_TSLA']
        macd_histogram = row['MACD_histogram_TSLA']
        volume_change = row['Volume Change_TSLA'] if not pd.isna(row['Volume Change_TSLA']) else 0
        atr = row['ATR_TSLA']
        lower_band = row['Lower Band_TSLA']
        upper_band = row['Upper Band_TSLA']
        stochastic_k = row['Stochastic_K_TSLA']
        stochastic_d = row['Stochastic_D_TSLA']
        obv = row['OBV_TSLA']
        obv_prev = data['OBV_TSLA'].iloc[i-1] if i > 0 else obv
        bb_width = row['BB_width_TSLA']
        rsi5 = row['RSI5_TSLA']
        macd_short = row['MACD_short_TSLA']
        macd_signal_short = row['MACD_signal_short_TSLA']
        vwap = row['VWAP_TSLA']

        target_tsll_weight, reasons = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma5, sma10, sma50, sma200, macd, macd_signal, macd_histogram,
            volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, rsi5, macd_short, macd_signal_short, vwap, current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        # 로그 기록
        logging.info(f"--- Date: {date} ---")
        logging.info(f"TSLA Close: ${row['Close_TSLA']:.2f}, TSLL Close: ${row['Close_TSLL']:.2f}")
        logging.info(f"Target TSLL Weight: {target_tsll_weight*100:.2f}%, Target TSLA Weight: {target_tsla_weight*100:.2f}%")
        if reasons:
            logging.info(f"Adjustment Reasons: {', '.join(reasons)}")
        else:
            logging.info("Adjustment Reasons: None")

        # 비중 변경 여부 확인 후 조정
        if abs(target_tsll_weight - prev_target_tsll_weight) > 0.01:  # 1% 이상 차이
            holdings, cash, trades = adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)
            if trades:
                logging.info("Trades Executed:")
                for trade in trades:
                    logging.info(f"  - {trade}")
            else:
                logging.info("Trades Executed: None (Insufficient cash)")
            prev_target_tsll_weight = target_tsll_weight
        else:
            logging.info("No adjustment needed (Target weight unchanged)")

        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0
        portfolio_values.append(total_value)
        dates.append(date)
        logging.info(f"Portfolio Value: ${total_value:.2f}, TSLL Weight: {current_tsll_weight*100:.2f}%, TSLA Shares: {holdings['TSLA']}, TSLL Shares: {holdings['TSLL']}, Cash: ${cash:.2f}")

    final_value = total_value
    final_tsll_weight = current_tsll_weight
    final_tsla_weight = (holdings['TSLA'] * data['Close_TSLA'].iloc[-1]) / final_value if final_value > 0 else 0
    cash_weight = cash / final_value if final_value > 0 else 0

    logging.info(f"Simulation ended. Final Portfolio Value: ${final_value:.2f}")
    logging.info(f"See {log_filename} for detailed logs.")
    return INITIAL_VALUE, final_value, holdings, final_tsll_weight, final_tsla_weight, cash, cash_weight, data['Close_TSLA'].iloc[-1], data['Close_TSLL'].iloc[-1], portfolio_values, dates

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Portfolio Simulation")
    parser.add_argument('--start_date', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, help="Number of days")

    args = parser.parse_args()
    params = load_params()

    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        end_date = start_date + timedelta(days=args.days if args.days else 180)
    else:
        if args.days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
        else:
            print("Please specify either --start_date or --days.")
            return

    initial_value, final_value, holdings, final_tsll_weight, final_tsla_weight, cash, cash_weight, tsla_close, tsll_close, portfolio_values, dates = simulate_portfolio(start_date, end_date, params)

    if initial_value is None:
        return

    print(f"\n### Portfolio Simulation Results ({args.days if args.days else 180} days)")
    print(f"- Simulation Period: {start_date.date()} to {end_date.date()}")
    print(f"- Initial Portfolio Value: ${initial_value:.2f}")
    print(f"- Final Portfolio Value: ${final_value:.2f}")
    print(f"- Portfolio Returns: {(final_value - initial_value) / initial_value:.2%}")

    print(f"\n### Current Stock Price ({end_date.date()})")
    print(f"- **TSLA Close**: ${tsla_close:.2f}")
    print(f"- **TSLL Close**: ${tsll_close:.2f}")

    table = [
        ["Date", start_date.date(), end_date.date()],
        ["Portfolio Value", f"${initial_value:.2f}", f"${final_value:.2f}"],
        ["TSLL Weight", "0.00%", f"{final_tsll_weight*100:.2f}%"],
        ["TSLA Weight", "0.00%", f"{final_tsla_weight*100:.2f}%"],
        ["Cash Weight", "100.00%", f"{cash_weight*100:.2f}%"],
        ["Cash Amount", f"${initial_value:.2f}", f"${cash:.2f}"],
        ["TSLA Shares", 0, holdings['TSLA']],
        ["TSLL Shares", 0, holdings['TSLL']]
    ]
    print("\n### Summary Table")
    print(tabulate(table, headers=["Metric", "Start", "End"], tablefmt="fancy_grid"))

    print("\n### Daily Portfolio Value Trend")
    value_table = [[dates[i], f"${portfolio_values[i]:.2f}"] for i in range(len(dates))]
    print(tabulate(value_table, headers=["Date", "Portfolio Value"], tablefmt="simple"))

if __name__ == "__main__":
    main()
