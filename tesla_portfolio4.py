#!/usr/bin/python3
import pandas as pd
import numpy as np
from itertools import product

# 백테스트 코드: 2년 과거 데이터 활용 (임시)

# 초기 포트폴리오 가치 설정
initial_portfolio_value = 100000

### 데이터 로드 및 전처리 ###
def load_and_prepare_data():
    # 데이터 로드
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    # 데이터 전처리
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 날짜 형식 통일 및 인덱스 설정
    fear_greed_df.set_index('date', inplace=True)
    tsla_df.set_index('Date', inplace=True)
    tsll_df.set_index('Date', inplace=True)

    # 데이터 병합
    data = pd.concat([tsla_df, tsll_df.add_prefix('TSLL_'), fear_greed_df['y']], axis=1).dropna()

    # 지표 계산
    data['Daily RSI'] = calculate_rsi(data['Close'], period=14)
    data['Weekly RSI'] = calculate_rsi(data['Close'].resample('W').last(), period=14).reindex(data.index, method='ffill')
    data['SMA50'] = calculate_sma(data['Close'], window=50)
    data['SMA200'] = calculate_sma(data['Close'], window=200)
    data['Upper Bollinger Band'], data['Lower Bollinger Band'] = calculate_bollinger_bands(data['Close'])
    data['Volume Change'] = data['Volume'].pct_change().fillna(0)

    return data

def calculate_rsi(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

### 백테스팅 함수 ###
def simulate_backtest(data, params):
    portfolio_value = initial_portfolio_value
    current_tsll_weight = 0.0

    for t in range(len(data) - 1):
        row = data.iloc[t]
        # 매수/매도 조건
        buy_conditions = [
            row['Daily RSI'] < params['daily_rsi_buy'],
            row['Weekly RSI'] < params['weekly_rsi_buy'],
            row['Close'] > row['Lower Bollinger Band'],
            row['Volume Change'] > params['volume_change_buy'],
            row['y'] < params['fg_buy']
        ]
        sell_conditions = [
            row['Daily RSI'] > params['daily_rsi_sell'],
            row['Weekly RSI'] > params['weekly_rsi_sell'],
            row['Close'] < row['Upper Bollinger Band'],
            row['Volume Change'] < params['volume_change_sell'],
            row['y'] > params['fg_sell']
        ]

        # 비중 조정
        base_weight = current_tsll_weight
        buy_adjustment = params['w_buy'] * sum(buy_conditions) * 0.1
        sell_adjustment = params['w_sell'] * sum(sell_conditions) * 0.1
        target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 0.8))
        current_tsll_weight = target_weight

        # 수익률 계산
        tsla_return = (data['Close'].iloc[t + 1] / data['Close'].iloc[t]) - 1
        tsll_return = (data['TSLL_Close'].iloc[t + 1] / data['TSLL_Close'].iloc[t]) - 1
        portfolio_return = current_tsll_weight * tsll_return + (1 - current_tsll_weight) * tsla_return
        portfolio_value *= (1 + portfolio_return)

    return portfolio_value

### 최적화 함수 ###
def optimize_parameters(data):
    best_value = 0
    best_params = {}

    # 파라미터 범위
    param_grid = {
        'daily_rsi_buy': [25, 30, 35],
        'daily_rsi_sell': [65, 70, 75],
        'weekly_rsi_buy': [35, 40, 45],
        'weekly_rsi_sell': [55, 60, 65],
        'fg_buy': [30, 40, 50],
        'fg_sell': [50, 60, 70],
        'volume_change_buy': [0.05, 0.1, 0.15],
        'volume_change_sell': [-0.15, -0.1, -0.05],
        'w_buy': [1.0, 1.5, 2.0],
        'w_sell': [1.0, 1.5, 2.0]
    }

    # 모든 조합 생성
    keys = list(param_grid.keys())
    combinations = product(*[param_grid[key] for key in keys])

    for combination in combinations:
        params = dict(zip(keys, combination))
        final_value = simulate_backtest(data, params)
        if final_value > best_value:
            best_value = final_value
            best_params = params

    return best_params, best_value

### 메인 함수 ###
def main():
    data = load_and_prepare_data()
    best_params, best_value = optimize_parameters(data)
    print("최적 파라미터:", best_params)
    print(f"최종 포트폴리오 가치: ${best_value:.2f}")

if __name__ == "__main__":
    main()
