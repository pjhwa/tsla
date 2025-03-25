import pandas as pd
import numpy as np
import json
from scipy.stats import linregress
import os
from datetime import datetime, timedelta
import argparse
from tabulate import tabulate

# 데이터 로드 및 전처리
def load_data(start_date, end_date):
    """지정된 기간의 주가 데이터와 공포탐욕지수를 로드하고 전처리합니다."""
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)
    data = data.dropna()

    data = data[(data.index >= start_date) & (data.index <= end_date)]

    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    data['Upper Band_TSLA'], _, data['Lower Band_TSLA'] = calculate_bollinger_bands(data['Close_TSLA'])
    data['MACD_TSLA'], data['MACD_signal_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']], 14)
    data['Weekly RSI_TSLA'] = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14).reindex(data.index, method='ffill').fillna(50)

    return data

# 지표 계산 함수
def calculate_rsi(series, timeperiod=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series, timeperiod=20):
    return series.rolling(window=timeperiod).mean()

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band, sma, lower_band

def calculate_atr(df, timeperiod=14):
    high_low = df['High_TSLA'] - df['Low_TSLA']
    high_close = np.abs(df['High_TSLA'] - df['Close_TSLA'].shift())
    low_close = np.abs(df['Low_TSLA'] - df['Close_TSLA'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    return atr

def get_rsi_trend(rsi_series, window=10):
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    if slope > 0.1:
        return "Increasing"
    elif slope < -0.1:
        return "Decreasing"
    return "Stable"

# 파라미터 로드
def load_params(file_path="optimal_params.json"):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{file_path} 파일이 없습니다. 기본 파라미터를 사용합니다.")
        return {
            "fg_buy": 25,
            "fg_sell": 75,
            "daily_rsi_buy": 30,
            "daily_rsi_sell": 70,
            "weekly_rsi_buy": 40,
            "weekly_rsi_sell": 60,
            "volume_change_strong_buy": 0.5,
            "volume_change_weak_buy": 0.2,
            "volume_change_sell": -0.2,
            "w_strong_buy": 2.0,
            "w_weak_buy": 1.0,
            "w_sell": 1.0
        }

# 비중 계산 로직
def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr, lower_band, upper_band, current_tsll_weight, params):
    base_weight = current_tsll_weight
    atr_normalized = atr / close if close > 0 else 0
    volume_change_strong_buy = params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = [
        fear_greed <= params["fg_buy"],
        daily_rsi < params["daily_rsi_buy"],
        (macd > macd_signal) and (macd_signal < 0),
        volume_change > volume_change_strong_buy,
        volume_change > volume_change_weak_buy,
        close < lower_band,
        (daily_rsi_trend == "Increasing") and (close > sma200)
    ]

    sell_conditions = [
        fear_greed >= params["fg_sell"],
        daily_rsi > params["daily_rsi_sell"],
        (macd < macd_signal) and (macd_signal > 0),
        volume_change < volume_change_sell,
        close > upper_band,
        (daily_rsi_trend == "Decreasing") and (close < sma200)
    ]

    strong_buy_count = buy_conditions[3]
    weak_buy_count = buy_conditions[4] and not strong_buy_count
    other_buy_count = sum(buy_conditions[:3] + buy_conditions[5:])
    sell_count = sum(sell_conditions)

    w_strong_buy = params["w_strong_buy"]
    w_weak_buy = params["w_weak_buy"]
    w_sell = params["w_sell"]

    buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * weak_buy_count + w_weak_buy * other_buy_count) * 0.1
    sell_adjustment = w_sell * sell_count * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    return target_weight, []

# 시뮬레이션 함수 (개선됨)
def simulate_portfolio(start_date, end_date, params):
    data = load_data(start_date, end_date)
    if data.empty:
        print("지정된 기간에 데이터가 없습니다.")
        return None, None, None, None, None, None, None

    initial_value = 100000
    cash = initial_value  # 현금 잔액
    holdings = {'TSLA': 0, 'TSLL': 0}  # 주식 보유량
    current_tsll_weight = 0.0

    for i in range(1, len(data)):
        row = data.iloc[i]
        fear_greed = row['y']
        daily_rsi = row['RSI_TSLA']
        weekly_rsi = row['Weekly RSI_TSLA']
        daily_rsi_trend = get_rsi_trend(data['RSI_TSLA'].iloc[max(0, i-10):i+1])
        close = row['Close_TSLA']
        sma50 = row['SMA50_TSLA']
        sma200 = row['SMA200_TSLA']
        macd = row['MACD_TSLA']
        macd_signal = row['MACD_signal_TSLA']
        volume_change = row['Volume Change_TSLA'] if not pd.isna(row['Volume Change_TSLA']) else 0
        atr = row['ATR_TSLA']
        lower_band = row['Lower Band_TSLA']
        upper_band = row['Upper Band_TSLA']

        # 목표 TSLL 비중 계산
        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr,
            lower_band, upper_band, current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        # 현재 포트폴리오 총 가치 계산
        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash

        # TSLA 조정
        required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])
        if required_tsla_shares > holdings['TSLA']:
            buy_shares = required_tsla_shares - holdings['TSLA']
            cost = buy_shares * row['Close_TSLA']
            if cash >= cost:
                cash -= cost
                holdings['TSLA'] += buy_shares
            else:
                buy_shares = int(cash / row['Close_TSLA'])
                cost = buy_shares * row['Close_TSLA']
                cash -= cost
                holdings['TSLA'] += buy_shares
        elif required_tsla_shares < holdings['TSLA']:
            sell_shares = holdings['TSLA'] - required_tsla_shares
            proceeds = sell_shares * row['Close_TSLA']
            cash += proceeds
            holdings['TSLA'] -= sell_shares

        # TSLL 조정
        required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])
        if required_tsll_shares > holdings['TSLL']:
            buy_shares = required_tsll_shares - holdings['TSLL']
            cost = buy_shares * row['Close_TSLL']
            if cash >= cost:
                cash -= cost
                holdings['TSLL'] += buy_shares
            else:
                buy_shares = int(cash / row['Close_TSLL'])
                cost = buy_shares * row['Close_TSLL']
                cash -= cost
                holdings['TSLL'] += buy_shares
        elif required_tsll_shares < holdings['TSLL']:
            sell_shares = holdings['TSLL'] - required_tsll_shares
            proceeds = sell_shares * row['Close_TSLL']
            cash += proceeds
            holdings['TSLL'] -= sell_shares

        # TSLL 비중 100% 보장: 남은 현금을 모두 TSLL에 투자 (단, 한 주 가격보다 적은 경우는 현금으로 유지)
        if target_tsll_weight == 1.0 and cash >= row['Close_TSLL']:
            buy_shares = int(cash / row['Close_TSLL'])
            if buy_shares > 0:
                cost = buy_shares * row['Close_TSLL']
                cash -= cost
                holdings['TSLL'] += buy_shares

        # 현재 TSLL 비중 업데이트
        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0

    # 최종 포트폴리오 가치 및 비중 계산
    final_value = holdings['TSLA'] * data['Close_TSLA'].iloc[-1] + holdings['TSLL'] * data['Close_TSLL'].iloc[-1] + cash
    final_tsll_weight = (holdings['TSLL'] * data['Close_TSLL'].iloc[-1]) / final_value if final_value > 0 else 0
    final_tsla_weight = (holdings['TSLA'] * data['Close_TSLA'].iloc[-1]) / final_value if final_value > 0 else 0
    cash_weight = cash / final_value if final_value > 0 else 0

    return initial_value, final_value, holdings, final_tsll_weight, final_tsla_weight, cash, cash_weight, data['Close_TSLA'].iloc[-1], data['Close_TSLL'].iloc[-1]

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="Portfolio Simulation")
    parser.add_argument('--start_date', type=str, help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, help="Number of days for simulation")

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

    initial_value, final_value, holdings, final_tsll_weight, final_tsla_weight, cash, cash_weight, tsla_close, tsll_close = simulate_portfolio(start_date, end_date, params)

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

if __name__ == "__main__":
    main()
