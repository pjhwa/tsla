import pandas as pd
import numpy as np
import json
import argparse
import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import linregress
import os
from tabulate import tabulate

# 초기 설정
initial_portfolio_value = 100000  # 초기 자산 $100,000

### 데이터 로드 함수 ###
def load_data(period):
    """지정된 기간에 해당하는 데이터를 로드합니다."""
    end_date = datetime.datetime.now()
    if period == '1M':
        start_date = end_date - relativedelta(months=1)
    elif period == '3M':
        start_date = end_date - relativedelta(months=3)
    elif period == '6M':
        start_date = end_date - relativedelta(months=6)
    elif period == '1Y':
        start_date = end_date - relativedelta(years=1)
    else:
        raise ValueError("유효하지 않은 기간입니다. '1M', '3M', '6M', '1Y' 중 선택하세요.")

    # 데이터 파일 로드
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    # Volume 열의 쉼표 제거 및 float 변환
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 지정된 기간으로 데이터 필터링
    fear_greed_df = fear_greed_df[(fear_greed_df['date'] >= start_date) & (fear_greed_df['date'] <= end_date)]
    tsla_df = tsla_df[(tsla_df['Date'] >= start_date) & (tsla_df['Date'] <= end_date)]
    tsll_df = tsll_df[(tsll_df['Date'] >= start_date) & (tsll_df['Date'] <= end_date)]

    return fear_greed_df, tsla_df, tsll_df

### 파라미터 로드 함수 ###
def load_params(file_path="optimal_params.json"):
    """JSON 파일에서 파라미터를 로드합니다."""
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
            "volume_change_buy": 0.1,
            "volume_change_sell": -0.1,
            "w_buy": 1.5,
            "w_sell": 1.0
        }

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
    rsi_series = rsi_series.dropna()
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    if slope > 0.1:
        return "Increasing"
    elif slope < -0.1:
        return "Decreasing"
    return "Stable"

### 포트폴리오 관리 함수 ###
def initialize_portfolio():
    """포트폴리오를 초기화합니다."""
    return {
        "cash": initial_portfolio_value,
        "TSLA": 0,
        "TSLL": 0
    }

def buy_stock(portfolio, stock, shares, price):
    """주식을 매수합니다."""
    cost = shares * price
    if portfolio["cash"] >= cost:
        portfolio["cash"] -= cost
        portfolio[stock] += shares
    else:
        shares_possible = int(portfolio["cash"] / price)
        portfolio["cash"] -= shares_possible * price
        portfolio[stock] += shares_possible

def sell_stock(portfolio, stock, shares, price):
    """주식을 매도합니다."""
    if portfolio[stock] >= shares:
        proceeds = shares * price
        portfolio["cash"] += proceeds
        portfolio[stock] -= shares
    else:
        proceeds = portfolio[stock] * price
        portfolio["cash"] += proceeds
        portfolio[stock] = 0

def get_portfolio_value(portfolio, tsla_price, tsll_price):
    """현재 포트폴리오 가치를 계산합니다."""
    return portfolio["cash"] + (portfolio["TSLA"] * tsla_price) + (portfolio["TSLL"] * tsll_price)

### 비중 계산 함수 ###
def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, lower_band, upper_band, current_tsll_weight, params):
    """TSLL의 목표 비중을 계산하고 조정 이유를 반환합니다."""
    base_weight = current_tsll_weight
    reasons = []

    buy_conditions = {
        f"Fear & Greed Index ≤ {params['fg_buy']}": fear_greed <= params["fg_buy"],
        f"Daily RSI < {params['daily_rsi_buy']}": daily_rsi < params["daily_rsi_buy"],
        "MACD > MACD Signal and MACD Signal < 0": (macd > macd_signal) and (macd_signal < 0),
        f"Volume Change > {params['volume_change_buy']}": volume_change > params["volume_change_buy"],
        "Close < Lower Bollinger Band": close < lower_band,
        "RSI Increasing and Close > SMA200": (rsi_trend == "Increasing") and (close > sma200)
    }

    sell_conditions = {
        f"Fear & Greed Index ≥ {params['fg_sell']}": fear_greed >= params["fg_sell"],
        f"Daily RSI > {params['daily_rsi_sell']}": daily_rsi > params["daily_rsi_sell"],
        "MACD < MACD Signal and MACD Signal > 0": (macd < macd_signal) and (macd_signal > 0),
        f"Volume Change < {params['volume_change_sell']}": volume_change < params["volume_change_sell"],
        "Close > Upper Bollinger Band": close > upper_band,
        "RSI Decreasing and Close < SMA200": (rsi_trend == "Decreasing") and (close < sma200)
    }

    buy_reasons = [condition for condition, is_true in buy_conditions.items() if is_true]
    sell_reasons = [condition for condition, is_true in sell_conditions.items() if is_true]

    w_buy = params["w_buy"]
    w_sell = params["w_sell"]
    buy_adjustment = w_buy * len(buy_reasons) * 0.1
    sell_adjustment = w_sell * len(sell_reasons) * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    reasons = []
    if buy_reasons:
        reasons.append("Buy Signals:")
        reasons.extend([f"  - {reason}" for reason in buy_reasons])
    if sell_reasons:
        reasons.append("Sell Signals:")
        reasons.extend([f"  - {reason}" for reason in sell_reasons])
    if not reasons:
        reasons.append("- No significant signals detected.")

    return target_weight, reasons

### 시뮬레이션 함수 ###
def simulate(period, opt_file):
    """지정된 기간 동안 포트폴리오 시뮬레이션을 실행합니다."""
    fear_greed_df, tsla_df, tsll_df = load_data(period)
    params = load_params(opt_file)

    portfolio = initialize_portfolio()
    log = []

    # 날짜 기준으로 데이터 정렬 및 병합 준비
    tsla_df.set_index('Date', inplace=True)
    tsll_df.set_index('Date', inplace=True)
    fear_greed_df.set_index('date', inplace=True)

    # 공통 날짜 추출
    common_dates = tsla_df.index.intersection(tsll_df.index).intersection(fear_greed_df.index)
    tsla_df = tsla_df.loc[common_dates]
    tsll_df = tsll_df.loc[common_dates]
    fear_greed_df = fear_greed_df.loc[common_dates]

    for date in common_dates:
        tsla_close = tsla_df.loc[date, 'Close']
        tsll_close = tsll_df.loc[date, 'Close']
        fear_greed = fear_greed_df.loc[date, 'y']

        # 지표 계산 (현재 날짜까지의 데이터만 사용)
        tsla_close_series = tsla_df['Close'][:date]
        daily_rsi = calculate_rsi(tsla_close_series, 14).iloc[-1] if len(tsla_close_series) >= 14 else 50
        weekly_rsi = calculate_rsi(tsla_close_series.resample('W').last(), 14).iloc[-1] if len(tsla_close_series.resample('W').last()) >= 14 else 50
        sma50 = calculate_sma(tsla_close_series, 50).iloc[-1] if len(tsla_close_series) >= 50 else tsla_close
        sma200 = calculate_sma(tsla_close_series, 200).iloc[-1] if len(tsla_close_series) >= 200 else tsla_close
        macd, macd_signal = calculate_macd(tsla_close_series)
        upper_band, _, lower_band = calculate_bollinger_bands(tsla_close_series)
        volume_change = tsla_df['Volume'][:date].pct_change().iloc[-1] if len(tsla_df['Volume'][:date]) > 1 else 0

        rsi_trend = get_rsi_trend(calculate_rsi(tsla_close_series, 14))

        # 현재 포트폴리오 가치 및 TSLL 비중 계산
        current_value = get_portfolio_value(portfolio, tsla_close, tsll_close)
        total_shares = portfolio["TSLA"] + portfolio["TSLL"]
        current_tsll_weight = portfolio["TSLL"] / total_shares if total_shares > 0 else 0

        # 목표 비중 계산
        target_tsll_weight, reasons = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, rsi_trend, tsla_close, sma50, sma200,
            macd.iloc[-1] if not macd.empty else 0, macd_signal.iloc[-1] if not macd_signal.empty else 0,
            volume_change, lower_band.iloc[-1] if not lower_band.empty else tsla_close,
            upper_band.iloc[-1] if not upper_band.empty else tsla_close,
            current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        # 필요한 주식 수 계산
        required_tsla_shares = int((target_tsla_weight * current_value) / tsla_close)
        required_tsll_shares = int((target_tsll_weight * current_value) / tsll_close)

        # 포트폴리오 조정
        if required_tsla_shares > portfolio["TSLA"]:
            buy_stock(portfolio, "TSLA", required_tsla_shares - portfolio["TSLA"], tsla_close)
        elif required_tsla_shares < portfolio["TSLA"]:
            sell_stock(portfolio, "TSLA", portfolio["TSLA"] - required_tsla_shares, tsla_close)

        if required_tsll_shares > portfolio["TSLL"]:
            buy_stock(portfolio, "TSLL", required_tsll_shares - portfolio["TSLL"], tsll_close)
        elif required_tsll_shares < portfolio["TSLL"]:
            sell_stock(portfolio, "TSLL", portfolio["TSLL"] - required_tsll_shares, tsll_close)

        # 로그 기록
        log.append({
            "date": date.strftime('%Y-%m-%d'),
            "portfolio_value": current_value,
            "tsla_shares": portfolio["TSLA"],
            "tsll_shares": portfolio["TSLL"],
            "cash": portfolio["cash"],
            "tsll_weight": target_tsll_weight
        })

    # 마지막 날의 데이터 추출
    last_date = common_dates[-1]
    tsla_close_last = tsla_df.loc[last_date, 'Close']
    tsll_close_last = tsll_df.loc[last_date, 'Close']
    fear_greed_last = fear_greed_df.loc[last_date, 'y']

    # 마지막 날의 지표 계산
    tsla_close_series_last = tsla_df['Close'][:last_date]
    daily_rsi_last = calculate_rsi(tsla_close_series_last, 14).iloc[-1] if len(tsla_close_series_last) >= 14 else 50
    weekly_rsi_last = calculate_rsi(tsla_close_series_last.resample('W').last(), 14).iloc[-1] if len(tsla_close_series_last.resample('W').last()) >= 14 else 50
    sma50_last = calculate_sma(tsla_close_series_last, 50).iloc[-1] if len(tsla_close_series_last) >= 50 else tsla_close_last
    sma200_last = calculate_sma(tsla_close_series_last, 200).iloc[-1] if len(tsla_close_series_last) >= 200 else tsla_close_last
    upper_band_last, _, lower_band_last = calculate_bollinger_bands(tsla_close_series_last)
    volume_change_last = tsla_df['Volume'][:last_date].pct_change().iloc[-1] if len(tsla_df['Volume'][:last_date]) > 1 else 0
    rsi_trend_last = get_rsi_trend(calculate_rsi(tsla_close_series_last, 14))

    # 시장 지표 테이블 출력
    indicators = [
        ["Fear & Greed Index", f"{fear_greed_last:.2f}", "-"],
        ["Daily RSI", f"{daily_rsi_last:.2f}", rsi_trend_last],
        ["Weekly RSI", f"{weekly_rsi_last:.2f}", "-"],
        ["TSLA Close", f"${tsla_close_last:.2f}", "-"],
        ["SMA50", f"${sma50_last:.2f}", "-"],
        ["SMA200", f"${sma200_last:.2f}", "-"],
        ["Upper Bollinger Band", f"${upper_band_last.iloc[-1]:.2f}" if not upper_band_last.empty else "-", "-"],
        ["Lower Bollinger Band", f"${lower_band_last.iloc[-1]:.2f}" if not lower_band_last.empty else "-", "-"],
        ["Volume Change", f"{volume_change_last:.2%}", "-"]
    ]
    print(f"\n### Market Indicators (as of {last_date.strftime('%Y-%m-%d')})")
    print(tabulate(indicators, headers=["Indicator", "Value", "Trend/Notes"], tablefmt="fancy_grid"))

    # 현재 주가 출력
    print(f"\n### Current Stock Prices ({last_date.strftime('%Y-%m-%d')})")
    print(f"- **TSLA Close**: ${tsla_close_last:.2f}")
    print(f"- **TSLL Close**: ${tsll_close_last:.2f}")

    # 마지막 날의 포트폴리오 상태 출력
    final_value = get_portfolio_value(portfolio, tsla_close_last, tsll_close_last)
    tsla_value = portfolio["TSLA"] * tsla_close_last
    tsll_value = portfolio["TSLL"] * tsll_close_last
    total_value = final_value
    tsla_weight = tsla_value / total_value if total_value > 0 else 0
    tsll_weight = tsll_value / total_value if total_value > 0 else 0
    returns = ((total_value - initial_portfolio_value) / initial_portfolio_value) * 100

    print(f"\n### Current Portfolio ({last_date.strftime('%Y-%m-%d')})")
    print(f"- Initial Investment: ${initial_portfolio_value:,.0f}")
    print(f"- TSLA: {portfolio['TSLA']} shares, value: ${tsla_value:.2f}")
    print(f"- TSLL: {portfolio['TSLL']} shares, value: ${tsll_value:.2f}")
    print(f"- Total Portfolio Value: ${total_value:.2f}")
    print(f"- TSLA Weight: {tsla_weight*100:.2f}%")
    print(f"- TSLL Weight: {tsll_weight*100:.2f}%")
    print(f"- Portfolio Returns: {returns:.2f}%")

    # 로그를 CSV로 저장
    log_df = pd.DataFrame(log)
    log_df.to_csv(f"simulation_log_{period}.csv", index=False)
    print(f"시뮬레이션 로그가 'simulation_log_{period}.csv'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="포트폴리오 시뮬레이션 프로그램")
    parser.add_argument('--period', type=str, choices=['1M', '3M', '6M', '1Y'], required=True, help="시뮬레이션 기간 (1M, 3M, 6M, 1Y)")
    parser.add_argument('--opt', type=str, default="optimal_params.json", help="최적 파라미터 JSON 파일 경로")

    args = parser.parse_args()
    simulate(args.period, args.opt)
