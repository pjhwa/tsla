import pandas as pd
import numpy as np
import json
from scipy.stats import linregress
import os
from datetime import datetime, timedelta
import argparse
from tabulate import tabulate

# 상수 정의
INITIAL_VALUE = 100000  # 초기 자산 $100,000

def load_and_preprocess_data(file_path, date_column='Date', volume_column='Volume'):
    """CSV 파일을 로드하고 전처리합니다."""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if volume_column in df.columns:
        df[volume_column] = df[volume_column].str.replace(',', '').astype(float)
    return df

def merge_dataframes(tsla_df, tsll_df, fear_greed_df, start_date, end_date):
    """데이터프레임을 'Date' 열을 기준으로 병합하고 지정된 기간으로 필터링합니다."""
    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.interpolate(method='linear')
    return data[(data.index >= start_date) & (data.index <= end_date)]

def calculate_rsi(series, timeperiod=14):
    """RSI(상대강도지수)를 계산합니다."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).interpolate(method='linear')

def calculate_sma(series, timeperiod=20):
    """단순 이동평균(SMA)을 계산합니다."""
    return series.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD, 신호선, 그리고 MACD Histogram을 계산합니다."""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    macd_histogram = macd - macd_signal  # MACD Histogram 계산
    return macd.interpolate(method='linear'), macd_signal.interpolate(method='linear'), macd_histogram.interpolate(method='linear')

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """볼린저 밴드를 계산합니다."""
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band.interpolate(method='linear'), sma.interpolate(method='linear'), lower_band.interpolate(method='linear')

def calculate_atr(df, timeperiod=14):
    """평균 진폭 범위(ATR)를 계산합니다."""
    high_low = df['High_TSLA'] - df['Low_TSLA']
    high_close = np.abs(df['High_TSLA'] - df['Close_TSLA'].shift())
    low_close = np.abs(df['Low_TSLA'] - df['Close_TSLA'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_stochastic(df, k_period=14, d_period=3):
    """스토캐스틱 오실레이터 %K와 %D를 계산합니다."""
    low_min = df['Low_TSLA'].rolling(window=k_period).min()
    high_max = df['High_TSLA'].rolling(window=k_period).max()
    k = 100 * ((df['Close_TSLA'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k.interpolate(method='linear'), d.interpolate(method='linear')

def calculate_obv(close, volume):
    """온밸런스 볼륨(OBV)을 계산합니다."""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index).interpolate(method='linear')

def get_rsi_trend(rsi_series, window=10):
    """RSI 추세를 분석합니다."""
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    return "Increasing" if slope > 0.1 else "Decreasing" if slope < -0.1 else "Stable"

def load_params(file_path="optimal_params.json"):
    """최적 파라미터를 로드하며 버전 관리와 기본값 제공합니다."""
    latest_version = "2.0"
    default_params = {
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
        "w_sell": 1.0,
        "stochastic_buy": 20,
        "stochastic_sell": 80,
        "obv_weight": 1.0,
        "bb_width_weight": 1.0
    }

    if not os.path.exists(file_path):
        print(f"경고: {file_path} 파일이 존재하지 않습니다. 기본 파라미터를 사용합니다.")
        return default_params

    try:
        with open(file_path, "r") as f:
            loaded_data = json.load(f)

        if "version" in loaded_data:
            file_version = loaded_data["version"]
            params = loaded_data["parameters"]
            if file_version != latest_version:
                print(f"경고: {file_path}의 버전({file_version})이 최신 버전({latest_version})과 다릅니다. 업데이트를 고려하세요.")
        else:
            params = loaded_data
            print(f"경고: {file_path}에 버전 정보가 없습니다. 최신 형식으로 업데이트를 권장합니다.")

        params.setdefault("stochastic_buy", 20)
        params.setdefault("stochastic_sell", 80)
        params.setdefault("obv_weight", 1.0)
        params.setdefault("bb_width_weight", 1.0)

        return params

    except (json.JSONDecodeError, KeyError) as e:
        print(f"경고: {file_path} 파일이 손상되었거나 형식이 잘못되었습니다. 기본 파라미터를 사용합니다. 오류: {e}")
        return default_params

def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, macd_histogram, volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, current_tsll_weight, params):
    """TSLL의 목표 비중을 계산합니다."""
    base_weight = current_tsll_weight
    atr_normalized = atr / close if close > 0 else 0
    volume_change_strong_buy = params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = [
        fear_greed <= params["fg_buy"],
        daily_rsi < params["daily_rsi_buy"],
        weekly_rsi < params["weekly_rsi_buy"],
        (macd > macd_signal) and (macd_signal < 0),
        macd_histogram > 0,  # MACD Histogram 매수 조건 추가
        volume_change > volume_change_strong_buy,
        volume_change > volume_change_weak_buy,
        close < lower_band,
        (daily_rsi_trend == "Increasing") and (close > sma200),
        stochastic_k < params["stochastic_buy"],
        obv > obv_prev,
        bb_width < 0.05
    ]

    sell_conditions = [
        fear_greed >= params["fg_sell"],
        daily_rsi > params["daily_rsi_sell"],
        weekly_rsi > params["weekly_rsi_sell"],
        (macd < macd_signal) and (macd_signal > 0),
        macd_histogram < 0,  # MACD Histogram 매도 조건 추가
        volume_change < volume_change_sell,
        close > upper_band,
        (daily_rsi_trend == "Decreasing") and (close < sma200),
        stochastic_k > params["stochastic_sell"],
        obv < obv_prev,
        bb_width > 0.15
    ]

    strong_buy_count = buy_conditions[5]  # volume_change > volume_change_strong_buy
    weak_buy_count = buy_conditions[6] and not strong_buy_count  # volume_change > volume_change_weak_buy
    other_buy_count = sum(buy_conditions[:4] + buy_conditions[7:11])  # 기타 매수 조건
    sell_count = sum(sell_conditions[:6])  # 기본 매도 조건
    obv_buy = buy_conditions[10]
    obv_sell = sell_conditions[9]
    bb_width_buy = buy_conditions[11]
    bb_width_sell = sell_conditions[10]

    w_strong_buy = params["w_strong_buy"]
    w_weak_buy = params["w_weak_buy"]
    w_sell = params["w_sell"]
    obv_weight = params["obv_weight"]
    bb_width_weight = params["bb_width_weight"]

    buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * weak_buy_count + w_weak_buy * other_buy_count + obv_weight * obv_buy + bb_width_weight * bb_width_buy) * 0.1
    sell_adjustment = (w_sell * sell_count + obv_weight * obv_sell + bb_width_weight * bb_width_sell) * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    return target_weight, []

def adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row):
    """포트폴리오를 목표 비중에 맞춰 조정합니다."""
    total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
    required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])
    required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])

    if required_tsla_shares > holdings['TSLA']:
        buy_shares = required_tsla_shares - holdings['TSLA']
        cost = buy_shares * row['Close_TSLA']
        if cash >= cost:
            cash -= cost
            holdings['TSLA'] += buy_shares
        else:
            buy_shares = int(cash / row['Close_TSLA'])
            cash -= buy_shares * row['Close_TSLA']
            holdings['TSLA'] += buy_shares
    elif required_tsla_shares < holdings['TSLA']:
        sell_shares = holdings['TSLA'] - required_tsla_shares
        proceeds = sell_shares * row['Close_TSLA']
        cash += proceeds
        holdings['TSLA'] -= sell_shares

    if required_tsll_shares > holdings['TSLL']:
        buy_shares = required_tsll_shares - holdings['TSLL']
        cost = buy_shares * row['Close_TSLL']
        if cash >= cost:
            cash -= cost
            holdings['TSLL'] += buy_shares
        else:
            buy_shares = int(cash / row['Close_TSLL'])
            cash -= buy_shares * row['Close_TSLL']
            holdings['TSLL'] += buy_shares
    elif required_tsll_shares < holdings['TSLL']:
        sell_shares = holdings['TSLL'] - required_tsll_shares
        proceeds = sell_shares * row['Close_TSLL']
        cash += proceeds
        holdings['TSLL'] -= sell_shares

    if target_tsll_weight == 1.0 and cash >= row['Close_TSLL']:
        buy_shares = int(cash / row['Close_TSLL'])
        if buy_shares > 0:
            cost = buy_shares * row['Close_TSLL']
            cash -= cost
            holdings['TSLL'] += buy_shares

    return holdings, cash

def load_data(start_date, end_date):
    """지정된 기간의 데이터를 로드하고 지표를 계산합니다."""
    tsla_df = load_and_preprocess_data('TSLA-history-2y.csv')
    tsll_df = load_and_preprocess_data('TSLL-history-2y.csv')
    fear_greed_df = load_and_preprocess_data('fear_greed_2years.csv', date_column='date', volume_column=None)
    data = merge_dataframes(tsla_df, tsll_df, fear_greed_df, start_date, end_date)

    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data['Close_TSLA'])
    data['Upper Band_TSLA'] = upper_band
    data['Middle Band_TSLA'] = middle_band
    data['Lower Band_TSLA'] = lower_band
    data['BB_width_TSLA'] = (upper_band - lower_band) / middle_band
    data['MACD_TSLA'], data['MACD_signal_TSLA'], data['MACD_histogram_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']], 14)
    data['Weekly RSI_TSLA'] = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14).reindex(data.index, method='ffill').fillna(50)
    data['Stochastic_K_TSLA'], data['Stochastic_D_TSLA'] = calculate_stochastic(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']])
    data['OBV_TSLA'] = calculate_obv(data['Close_TSLA'], data['Volume_TSLA'])

    return data

def simulate_portfolio(start_date, end_date, params):
    """지정된 기간 동안 포트폴리오 시뮬레이션을 수행합니다."""
    data = load_data(start_date, end_date)
    if data.empty:
        print("지정된 기간에 데이터가 없습니다.")
        return None, None, None, None, None, None, None

    cash = INITIAL_VALUE
    holdings = {'TSLA': 0, 'TSLL': 0}
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
        macd_histogram = row['MACD_histogram_TSLA']  # MACD Histogram 값 추출
        volume_change = row['Volume Change_TSLA'] if not pd.isna(row['Volume Change_TSLA']) else 0
        atr = row['ATR_TSLA']
        lower_band = row['Lower Band_TSLA']
        upper_band = row['Upper Band_TSLA']
        stochastic_k = row['Stochastic_K_TSLA']
        stochastic_d = row['Stochastic_D_TSLA']
        obv = row['OBV_TSLA']
        obv_prev = data['OBV_TSLA'].iloc[i-1] if i > 0 else obv
        bb_width = row['BB_width_TSLA']

        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, macd_histogram,
            volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        holdings, cash = adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)
        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0

    final_value = holdings['TSLA'] * data['Close_TSLA'].iloc[-1] + holdings['TSLL'] * data['Close_TSLL'].iloc[-1] + cash
    final_tsll_weight = (holdings['TSLL'] * data['Close_TSLL'].iloc[-1]) / final_value if final_value > 0 else 0
    final_tsla_weight = (holdings['TSLA'] * data['Close_TSLA'].iloc[-1]) / final_value if final_value > 0 else 0
    cash_weight = cash / final_value if final_value > 0 else 0

    return INITIAL_VALUE, final_value, holdings, final_tsll_weight, final_tsla_weight, cash, cash_weight, data['Close_TSLA'].iloc[-1], data['Close_TSLL'].iloc[-1]

def main():
    """메인 함수: 명령줄 인자를 파싱하고 시뮬레이션을 실행합니다."""
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
