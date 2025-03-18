#!/usr/bin/python3
import pandas as pd
import numpy as np

# 백테스트 결과 디버깅

# 초기 설정
initial_portfolio_value = 100000  # 초기 자산 $100,000

### 데이터 로드 함수 ###
def load_data():
    tsla_df = pd.read_csv('TSLA-history-2y.csv', index_col='Date', parse_dates=True)
    tsll_df = pd.read_csv('TSLL-history-2y.csv', index_col='Date', parse_dates=True)
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', index_col='date', parse_dates=True)

    # Volume 열 처리
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 오름차순 정렬
    tsla_df = tsla_df.sort_index(ascending=True)
    tsll_df = tsll_df.sort_index(ascending=True)
    fear_greed_df = fear_greed_df.sort_index(ascending=True)

    # 공통 날짜 필터링
    common_dates = tsla_df.index.intersection(tsll_df.index).intersection(fear_greed_df.index)
    tsla_df = tsla_df.loc[common_dates]
    tsll_df = tsll_df.loc[common_dates]
    fear_greed_series = fear_greed_df['y'].loc[common_dates]

    print(f"Debug: Common dates found: {len(common_dates)} days from {common_dates.min()} to {common_dates.max()}")
    return tsla_df, tsll_df, fear_greed_series

### 지표 계산 함수 ###
def calculate_rsi(series, timeperiod=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series, timeperiod=50):
    return series.rolling(window=timeperiod).mean()

def add_indicators(tsla_df):
    tsla_df['RSI'] = calculate_rsi(tsla_df['Close'])
    tsla_df['SMA50'] = calculate_sma(tsla_df['Close'])
    print(f"Debug: Before dropna - Rows: {len(tsla_df)}, Date Range: {tsla_df.index.min()} to {tsla_df.index.max()}")
    tsla_df = tsla_df.dropna()
    print(f"Debug: After dropna - Rows: {len(tsla_df)}, Date Range: {tsla_df.index.min()} to {tsla_df.index.max()}")
    return tsla_df

### 백테스팅 함수 ###
def simulate_backtest(tsla_df, tsll_df, fear_greed_series, w_buy, w_sell):
    portfolio_value = initial_portfolio_value
    current_tsll_weight = 0.0

    for t in range(len(tsla_df) - 1):
        current_date = tsla_df.index[t]
        next_date = tsla_df.index[t + 1]

        fear_greed = fear_greed_series.loc[current_date]
        daily_rsi = tsla_df['RSI'].iloc[t]
        close = tsla_df['Close'].iloc[t]
        sma50 = tsla_df['SMA50'].iloc[t]

        # 매수/매도 조건
        buy_conditions = [fear_greed <= 30, daily_rsi < 35, close > sma50]
        sell_conditions = [fear_greed >= 70, daily_rsi > 65, close < sma50]

        base_weight = current_tsll_weight
        buy_adjustment = w_buy * sum(buy_conditions) * 0.1
        sell_adjustment = w_sell * sum(sell_conditions) * 0.1
        target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))
        adjustment = target_weight - current_tsll_weight
        current_tsll_weight = target_weight
        tsla_weight = 1 - current_tsll_weight

        tsla_return = (tsla_df['Close'].iloc[t + 1] / tsla_df['Close'].iloc[t]) - 1
        tsll_return = (tsll_df['Close'].iloc[t + 1] / tsll_df['Close'].iloc[t]) - 1
        portfolio_return = current_tsll_weight * tsll_return + tsla_weight * tsla_return
        portfolio_value *= (1 + portfolio_return)

        # 디버깅 출력
        print(f"Date: {current_date.strftime('%Y-%m-%d')}, "
              f"Fear&Greed: {fear_greed:.2f}, RSI: {daily_rsi:.2f}, Close: {close:.2f}, SMA50: {sma50:.2f}, "
              f"Buy Conditions: {sum(buy_conditions)}/3, Sell Conditions: {sum(sell_conditions)}/3, "
              f"Portfolio Value: ${portfolio_value:.2f}, TSLA Weight: {tsla_weight:.2%}, TSLL Weight: {current_tsll_weight:.2%}, "
              f"Adjustment: {adjustment:.2%}")

    return portfolio_value

### 메인 실행 함수 ###
def main():
    print("\n=== 전체 기간 백테스팅 시작 ===")
    tsla_df, tsll_df, fear_greed_series = load_data()
    tsla_df = add_indicators(tsla_df)
    final_value = simulate_backtest(tsla_df, tsll_df, fear_greed_series, w_buy=2.0, w_sell=2.0)

    print("\n### 백테스팅 결과")
    print(f"- 최종 포트폴리오 가치: ${final_value:.2f}")
    print(f"- 총 수익률: {((final_value / initial_portfolio_value) - 1) * 100:.2f}%")

if __name__ == "__main__":
    main()
