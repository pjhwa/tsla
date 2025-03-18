#!/usr/bin/python3
import pandas as pd
import numpy as np
from scipy.stats import linregress

# 초기 설정
initial_portfolio_value = 100000  # 초기 자산 $100,000

### 데이터 로드 함수 ###
def load_data():
    # TSLA와 TSLL 가격 데이터 로드
    tsla_df = pd.read_csv('TSLA-historicaldata.csv', index_col='Date', parse_dates=True)
    tsll_df = pd.read_csv('TSLL-historicaldata.csv', index_col='Date', parse_dates=True)
    fear_greed_df = pd.read_csv('fear_greed_1year.csv', index_col='date', parse_dates=True)

    # Volume 열의 쉼표 제거 및 숫자형으로 변환
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 공통 날짜 찾기
    common_dates = tsla_df.index.intersection(tsll_df.index).intersection(fear_greed_df.index)
    tsla_df = tsla_df.loc[common_dates]
    tsll_df = tsll_df.loc[common_dates]
    fear_greed_series = fear_greed_df['y'].loc[common_dates]

    return tsla_df, tsll_df, fear_greed_series

### 지표 계산 함수 ###
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

def calculate_rsi_trend(rsi_window):
    if len(rsi_window) < 10:
        return "Stable"
    slope, _, _, _, _ = linregress(range(10), rsi_window[-10:])
    if slope > 0.1:
        return "Increasing"
    elif slope < -0.1:
        return "Decreasing"
    return "Stable"

def add_indicators(tsla_df):
    # 기존 지표 계산
    tsla_df['RSI'] = calculate_rsi(tsla_df['Close'], timeperiod=14)
    tsla_df['SMA50'] = calculate_sma(tsla_df['Close'], timeperiod=50)
    tsla_df['SMA200'] = calculate_sma(tsla_df['Close'], timeperiod=200)
    tsla_df['MACD'], tsla_df['MACD_signal'] = calculate_macd(tsla_df['Close'])
    tsla_df['Upper Band'], tsla_df['Middle Band'], tsla_df['Lower Band'] = calculate_bollinger_bands(tsla_df['Close'])
    tsla_df['Volume Change'] = tsla_df['Volume'].pct_change()

    # RSI 추세 계산 함수 정의
    def get_rsi_trend(window):
        # 창 크기가 10보다 작으면 기본값 "Stable" 반환
        if len(window) < 10:
            return "Stable"
        # 최근 10개 데이터로 선형 회귀를 통해 기울기 계산
        slope, _, _, _, _ = linregress(range(10), window[-10:])
        if slope > 0.1:
            return "Increasing"
        elif slope < -0.1:
            return "Decreasing"
        return "Stable"

    # RSI_trend 계산 (리스트 컴프리헨션 사용)
    rsi_trend_list = [get_rsi_trend(tsla_df['RSI'].iloc[max(0, i-10):i+1])
                      for i in range(len(tsla_df))]
    tsla_df['RSI_trend'] = rsi_trend_list

    # 결측값 제거
    return tsla_df.dropna()

# 참고: calculate_rsi, calculate_sma, calculate_macd, calculate_bollinger_bands 함수는
# 기존에 정의된 것으로 가정합니다.

### 백테스팅 함수 ###
def simulate_backtest(tsla_df, tsll_df, fear_greed_series, w_buy, w_sell):
    portfolio_value = initial_portfolio_value
    current_tsll_weight = 0.0

    for t in range(len(tsla_df) - 1):
        current_date = tsla_df.index[t]
        next_date = tsla_df.index[t + 1]

        # 현재 날짜의 지표
        fear_greed = fear_greed_series.loc[current_date]
        daily_rsi = tsla_df['RSI'].iloc[t]
        close = tsla_df['Close'].iloc[t]
        sma200 = tsla_df['SMA200'].iloc[t]
        macd = tsla_df['MACD'].iloc[t]
        macd_signal = tsla_df['MACD_signal'].iloc[t]
        volume_change = tsla_df['Volume Change'].iloc[t]
        lower_band = tsla_df['Lower Band'].iloc[t]
        upper_band = tsla_df['Upper Band'].iloc[t]
        rsi_trend = tsla_df['RSI_trend'].iloc[t]

        # 매수 및 매도 조건
        buy_conditions = [
            fear_greed <= 25,                    # 극단적 공포
            daily_rsi < 30,                      # 과매도
            (macd > macd_signal) and (macd_signal < 0),  # MACD 매수 신호
            volume_change > 0.1,                 # 거래량 10% 증가
            close < lower_band,                  # 하단 볼린저 밴드 이하
            (rsi_trend == "Increasing") and (close > sma200)  # 상승 추세
        ]
        sell_conditions = [
            fear_greed >= 75,                    # 극단적 탐욕
            daily_rsi > 70,                      # 과매수
            (macd < macd_signal) and (macd_signal > 0),  # MACD 매도 신호
            volume_change < -0.1,                # 거래량 10% 감소
            close > upper_band,                  # 상단 볼린저 밴드 이상
            (rsi_trend == "Decreasing") and (close < sma200)  # 하락 추세
        ]

        # 비중 조정 계산
        base_weight = current_tsll_weight
        buy_adjustment = w_buy * sum(buy_conditions) * 0.1
        sell_adjustment = w_sell * sum(sell_conditions) * 0.1
        target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))
        current_tsll_weight = target_weight

        # 다음 날 수익률 계산
        tsla_return = (tsla_df['Close'].iloc[t + 1] / tsla_df['Close'].iloc[t]) - 1
        tsll_return = (tsll_df['Close'].iloc[t + 1] / tsll_df['Close'].iloc[t]) - 1
        portfolio_return = current_tsll_weight * tsll_return + (1 - current_tsll_weight) * tsla_return
        portfolio_value *= (1 + portfolio_return)

    return portfolio_value

### 최적화 함수 ###
def optimize_weights(tsla_df, tsll_df, fear_greed_series):
    best_value = 0
    best_params = (0, 0)

    # w_buy와 w_sell에 대해 그리드 서치 (0부터 2까지 0.1 단위)
    for w_buy in np.arange(0, 2.1, 0.1):
        for w_sell in np.arange(0, 2.1, 0.1):
            final_value = simulate_backtest(tsla_df, tsll_df, fear_greed_series, w_buy, w_sell)
            if final_value > best_value:
                best_value = final_value
                best_params = (w_buy, w_sell)

    return best_params, best_value

### 메인 실행 함수 ###
def main():
    # 데이터 로드
    print("데이터 로드 중...")
    tsla_df, tsll_df, fear_greed_series = load_data()

    # 지표 계산
    print("지표 계산 중...")
    tsla_df = add_indicators(tsla_df)

    # 백테스팅 및 최적화
    print("백테스팅 및 가중치 최적화 중...")
    (w_buy, w_sell), final_value = optimize_weights(tsla_df, tsll_df, fear_greed_series)

    # 결과 출력
    print("\n### 최적화 결과")
    print(f"- 최적 매수 가중치 (w_buy): {w_buy:.1f}")
    print(f"- 최적 매도 가중치 (w_sell): {w_sell:.1f}")
    print(f"- 최종 포트폴리오 가치: ${final_value:.2f}")
    print(f"- 총 수익률: {((final_value / initial_portfolio_value) - 1) * 100:.2f}%")

if __name__ == "__main__":
    main()
