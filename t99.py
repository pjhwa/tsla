#!/usr/bin/python3
import yfinance as yf
import pandas as pd
import requests
import datetime
import numpy as np
from scipy.stats import linregress
from tabulate import tabulate
import json
import csv

# 초기 설정
portfolio_value = 100000  # 초기 자산 $100,000
current_tsll_weight = 0.0  # TSLL 비중 0%로 시작

### 최적 파라미터 로드 함수 ###
def load_optimal_params(file_path="optimal_params.json"):
    """JSON 파일에서 최적 파라미터를 로드합니다. 파일이 없으면 기본값을 반환합니다."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("최적 파라미터 파일이 없습니다. 기본값을 사용합니다.")
        return {
            "fg_buy": 25,
            "fg_sell": 75,
            "daily_rsi_buy": 30,
            "daily_rsi_sell": 70,
            "weekly_rsi_buy": 40,
            "weekly_rsi_sell": 60,
            "volume_change_buy": 0.1,
            "volume_change_sell": -0.1,
            "w_buy": 1.5,
            "w_sell": 1.0
        }

### 데이터 수집 함수 ###
def get_fear_greed_index():
    """CNN의 Fear & Greed Index를 가져옵니다."""
    current_date = datetime.date.today()
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{current_date.strftime('%Y-%m-%d')}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
    response = requests.get(url, headers=headers, timeout=10)
    data = response.json()
    fg_df = pd.DataFrame(data["fear_and_greed_historical"]["data"])
    fg_df['x'] = pd.to_datetime(fg_df['x'], unit='ms').dt.date
    fg_df = fg_df.rename(columns={'x': 'Date', 'y': 'Fear & Greed Index'})
    return fg_df['Fear & Greed Index'].iloc[-1]

def get_stock_data(ticker, period="max", interval="1d"):
    """주식 데이터를 가져와 기술적 지표를 계산합니다."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df = df.dropna(subset=['Close'])
        df['Close'] = df['Close'].round(2)
        df['RSI'] = calculate_rsi(df['Close'], timeperiod=14)
        df['SMA50'] = calculate_sma(df['Close'], timeperiod=50)
        df['SMA200'] = calculate_sma(df['Close'], timeperiod=200)
        df['Upper Band'], df['Middle Band'], df['Lower Band'] = calculate_bollinger_bands(df['Close'])
        df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
        df['Volume Change'] = df['Volume'].pct_change()
        return df
    except Exception as e:
        print(f"{ticker} 데이터 로드 오류: {e}")
        return None

def get_weekly_rsi(ticker, period="max"):
    """주간 RSI를 계산합니다."""
    try:
        weekly_df = yf.Ticker(ticker).history(period=period, interval="1wk")
        weekly_df = weekly_df.dropna(subset=['Close'])
        weekly_df['Close'] = weekly_df['Close'].round(2)
        weekly_df['RSI'] = calculate_rsi(weekly_df['Close'], timeperiod=14)
        return weekly_df['RSI'].iloc[-1]
    except Exception as e:
        print(f"{ticker} 주간 RSI 계산 오류: {e}")
        return None

### 지표 계산 함수 ###
def calculate_rsi(series, timeperiod=14):
    """RSI(상대강도지수)를 계산합니다."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series, timeperiod=20):
    """단순 이동평균(SMA)을 계산합니다."""
    return series.rolling(window=timeperiod).mean()

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD와 신호선을 계산합니다."""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """볼린저 밴드를 계산합니다."""
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band, sma, lower_band

def get_rsi_trend(rsi_series, window=10):
    """RSI의 추세를 분석합니다."""
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    if slope > 0.1:
        return "Increasing"
    elif slope < -0.1:
        return "Decreasing"
    return "Stable"

### 전략 및 포트폴리오 관리 ###
def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, lower_band, upper_band, optimal_params):
    """TSLL의 목표 비중을 계산합니다."""
    base_weight = current_tsll_weight
    reasons = []

    # 매수 조건 정의
    buy_conditions = {
        f"Fear & Greed Index ≤ {optimal_params['fg_buy']}": fear_greed <= optimal_params["fg_buy"],
        f"Daily RSI < {optimal_params['daily_rsi_buy']}": daily_rsi < optimal_params["daily_rsi_buy"],
        "MACD > MACD Signal and MACD Signal < 0": (macd > macd_signal) and (macd_signal < 0),
        f"Volume Change > {optimal_params['volume_change_buy']}": volume_change > optimal_params["volume_change_buy"],
        "Close < Lower Bollinger Band": close < lower_band,
        f"RSI Increasing and Close > SMA200": (daily_rsi_trend == "Increasing") and (close > sma200)
    }

    # 매도 조건 정의
    sell_conditions = {
        f"Fear & Greed Index ≥ {optimal_params['fg_sell']}": fear_greed >= optimal_params["fg_sell"],
        f"Daily RSI > {optimal_params['daily_rsi_sell']}": daily_rsi > optimal_params["daily_rsi_sell"],
        "MACD < MACD Signal and MACD Signal > 0": (macd < macd_signal) and (macd_signal > 0),
        f"Volume Change < {optimal_params['volume_change_sell']}": volume_change < optimal_params["volume_change_sell"],
        "Close > Upper Bollinger Band": close > upper_band,
        f"RSI Decreasing and Close < SMA200": (daily_rsi_trend == "Decreasing") and (close < sma200)
    }

    # 충족된 조건 기록
    buy_reasons = [condition for condition, is_true in buy_conditions.items() if is_true]
    sell_reasons = [condition for condition, is_true in sell_conditions.items() if is_true]

    # 비중 조정
    w_buy = optimal_params["w_buy"]
    w_sell = optimal_params["w_sell"]
    buy_adjustment = w_buy * len(buy_reasons) * 0.1
    sell_adjustment = w_sell * len(sell_reasons) * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    # 조정 이유 기록
    if buy_reasons:
        reasons.append(f"Buy Signal: {', '.join(buy_reasons)}")
    if sell_reasons:
        reasons.append(f"Sell Signal: {', '.join(sell_reasons)}")

    return target_weight, reasons

def adjust_portfolio(target_tsll_weight):
    """포트폴리오 비중을 조정합니다."""
    global current_tsll_weight
    if current_tsll_weight < target_tsll_weight:
        buy_amount = (target_tsll_weight - current_tsll_weight) * portfolio_value
        print(f" - Buy TSLL: ${buy_amount:.2f}")
    elif current_tsll_weight > target_tsll_weight:
        sell_amount = (current_tsll_weight - target_tsll_weight) * portfolio_value
        print(f" - Sell TSLL: ${sell_amount:.2f}")
    current_tsll_weight = target_tsll_weight

### 로깅 함수 ###
def log_decision(date, target_tsll_weight, reasons):
    """조정 내역을 CSV 파일에 기록합니다."""
    with open("portfolio_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date, target_tsll_weight, "; ".join(reasons)])

### 실행 함수 ###
def analyze_and_recommend():
    """데이터를 분석하고 포트폴리오 조정을 추천합니다."""
    print("데이터 로드 중...")
    optimal_params = load_optimal_params()
    fear_greed = get_fear_greed_index()
    tsla_df = get_stock_data("TSLA")
    tsll_df = get_stock_data("TSLL")
    weekly_rsi = get_weekly_rsi("TSLA")

    if tsla_df is None or tsll_df is None or weekly_rsi is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return

    # TSLA 및 TSLL의 현재 주가
    tsla_close = tsla_df['Close'].iloc[-1]
    tsll_close = tsll_df['Close'].iloc[-1]

    # 데이터 기준 날짜
    data_date = tsla_df.index[-1].strftime('%Y-%m-%d')

    # TSLA 지표
    daily_rsi = tsla_df['RSI'].iloc[-1]
    daily_rsi_trend = get_rsi_trend(tsla_df['RSI'].tail(10))
    macd = tsla_df['MACD'].iloc[-1]
    macd_signal = tsla_df['MACD_signal'].iloc[-1]
    close = tsla_df['Close'].iloc[-1]
    sma50 = tsla_df['SMA50'].iloc[-1]
    sma200 = tsla_df['SMA200'].iloc[-1]
    upper_band = tsla_df['Upper Band'].iloc[-1]
    lower_band = tsla_df['Lower Band'].iloc[-1]
    volume_change = tsla_df['Volume Change'].iloc[-1]

    # 시장 지표 테이블
    indicators = [
        ["Fear & Greed Index", f"{fear_greed:.2f}", "-"],
        ["Daily RSI", f"{daily_rsi:.2f}", daily_rsi_trend],
        ["Weekly RSI", f"{weekly_rsi:.2f}", "-"],
        ["TSLA Close", f"${close:.2f}", "-"],
        ["SMA50", f"${sma50:.2f}", "-"],
        ["SMA200", f"${sma200:.2f}", "-"],
        ["Upper Bollinger Band", f"${upper_band:.2f}", "-"],
        ["Lower Bollinger Band", f"${lower_band:.2f}", "-"],
        ["Volume Change", f"{volume_change:.2%}", "-"]
    ]
    print(f"\n### Market Indicators (as of {data_date})")
    print(tabulate(indicators, headers=["Indicator", "Value", "Trend/Notes"], tablefmt="fancy_grid"))

    # 현재 주가 표시
    print("\n### Current Stock Prices")
    print(f"- **TSLA Close**: ${tsla_close:.2f}")
    print(f"- **TSLL Close**: ${tsll_close:.2f}")

    # 포트폴리오 비중 계산
    target_tsll_weight, reasons = get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, lower_band, upper_band, optimal_params)
    target_tsla_weight = 1 - target_tsll_weight

    # 추천 비중 표시
    print("\n### Recommended Portfolio Weights")
    print(f"- **TSLA Weight**: {target_tsla_weight*100:.0f}%")
    print(f"- **TSLL Weight**: {target_tsll_weight*100:.0f}%")

    # 포트폴리오 조정
    print("\n### Portfolio Adjustment")
    adjust_portfolio(target_tsll_weight)

    # 비중 조정 이유
    print("\n### Adjustment Reasons")
    if reasons:
        for reason in reasons:
            print(f"- {reason}")
    else:
        print("- No adjustment needed based on current conditions.")

    # 로그 기록
    log_decision(data_date, target_tsll_weight, reasons)

if __name__ == "__main__":
    analyze_and_recommend()
