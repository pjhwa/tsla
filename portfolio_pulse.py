import yfinance as yf
import pandas as pd
import requests
import datetime
import numpy as np
from scipy.stats import linregress
from tabulate import tabulate
import json
import csv
import os

# 초기 설정
initial_portfolio_value = 100000  # 초기 자산 $100,000

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
            "volume_change_strong_buy": 0.5,  # 강한 Buy: 50% 이상 증가
            "volume_change_weak_buy": 0.2,    # 약한 Buy: 20% 이상 증가
            "volume_change_sell": -0.2,       # Sell: 20% 이상 감소
            "w_strong_buy": 2.0,              # 강한 Buy 가중치
            "w_weak_buy": 1.0,                # 약한 Buy 가중치
            "w_sell": 1.0                     # Sell 가중치
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
        df['ATR'] = calculate_atr(df, timeperiod=14)  # ATR 추가
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

def calculate_atr(df, timeperiod=14):
    """ATR(Average True Range)을 계산합니다."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    return atr

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

### transactions.txt 파일 처리 함수 ###
def load_transactions(file_path="transactions.txt"):
    """transactions.txt 파일을 읽어 현재 보유 주식 수와 초기 투자액을 계산합니다."""
    if not os.path.exists(file_path):
        print("transactions.txt 파일이 없습니다. 초기 자산으로 계산을 진행합니다.")
        return {}, 0  # 보유 주식, 초기 투자액

    try:
        transactions_df = pd.read_csv(file_path, sep='\s+', names=["date", "ticker", "action", "shares", "stock_price"])
        holdings = {}
        initial_investment = 0

        for index, row in transactions_df.iterrows():
            ticker = row['ticker']
            action = row['action']
            shares = row['shares']
            price = row['stock_price']

            if index == 0 and action == "hold":
                initial_investment = shares * price
                holdings[ticker] = shares
            elif action == "buy":
                holdings[ticker] = holdings.get(ticker, 0) + shares
            elif action == "sell":
                if ticker in holdings and holdings[ticker] >= shares:
                    holdings[ticker] -= shares
                    if holdings[ticker] == 0:
                        del holdings[ticker]
                else:
                    print(f"Insufficient shares to sell: {ticker}, {shares} shares")

        current_holdings = {ticker: shares for ticker, shares in holdings.items() if shares > 0}
        return current_holdings, initial_investment
    except Exception as e:
        print(f"transactions.txt 파일 로드 오류: {e}")
        return {}, 0

### 포트폴리오 가치 및 수익률 계산 ###
def calculate_portfolio_metrics(current_holdings, tsla_close, tsll_close, initial_investment):
    """현재 포트폴리오 가치, 비중, 수익률을 계산합니다."""
    tsla_shares = current_holdings.get("TSLA", 0)
    tsll_shares = current_holdings.get("TSLL", 0)
    tsla_value = tsla_shares * tsla_close
    tsll_value = tsll_shares * tsll_close
    total_value = tsla_value + tsll_value
    tsla_weight = tsla_value / total_value if total_value > 0 else 0
    tsll_weight = tsll_value / total_value if total_value > 0 else 0
    returns = ((total_value - initial_investment) / initial_investment * 100) if initial_investment > 0 else 0
    return total_value, tsla_value, tsll_value, tsla_weight, tsll_weight, returns

### 전략 및 비중 계산 ###
def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr, lower_band, upper_band, current_tsll_weight, optimal_params):
    """TSLL의 목표 비중을 계산하고 조정 이유를 반환합니다."""
    base_weight = current_tsll_weight
    reasons = []

    # 동적 임계값 계산 (ATR 기반)
    atr_normalized = atr / close  # 가격 대비 변동성 비율
    volume_change_strong_buy = optimal_params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = optimal_params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = optimal_params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = {
        f"Fear & Greed Index ≤ {optimal_params['fg_buy']}": fear_greed <= optimal_params["fg_buy"],
        f"Daily RSI < {optimal_params['daily_rsi_buy']}": daily_rsi < optimal_params["daily_rsi_buy"],
        "MACD > MACD Signal and MACD Signal < 0": (macd > macd_signal) and (macd_signal < 0),
        f"Volume Change > {volume_change_strong_buy:.2f} (Strong Buy)": volume_change > volume_change_strong_buy,
        f"Volume Change > {volume_change_weak_buy:.2f} (Weak Buy)": volume_change > volume_change_weak_buy,
        "Close < Lower Bollinger Band": close < lower_band,
        f"RSI Increasing and Close > SMA200": (daily_rsi_trend == "Increasing") and (close > sma200)
    }

    sell_conditions = {
        f"Fear & Greed Index ≥ {optimal_params['fg_sell']}": fear_greed >= optimal_params["fg_sell"],
        f"Daily RSI > {optimal_params['daily_rsi_sell']}": daily_rsi > optimal_params["daily_rsi_sell"],
        "MACD < MACD Signal and MACD Signal > 0": (macd < macd_signal) and (macd_signal > 0),
        f"Volume Change < {volume_change_sell:.2f}": volume_change < volume_change_sell,
        "Close > Upper Bollinger Band": close > upper_band,
        f"RSI Decreasing and Close < SMA200": (daily_rsi_trend == "Decreasing") and (close < sma200)
    }

    buy_reasons = [condition for condition, is_true in buy_conditions.items() if is_true]
    sell_reasons = [condition for condition, is_true in sell_conditions.items() if is_true]

    # 세분화된 가중치 적용
    w_strong_buy = optimal_params["w_strong_buy"]
    w_weak_buy = optimal_params["w_weak_buy"]
    w_sell = optimal_params["w_sell"]

    strong_buy_count = sum(1 for r in buy_reasons if "Strong Buy" in r)
    weak_buy_count = sum(1 for r in buy_reasons if "Weak Buy" in r and "Strong Buy" not in r)
    other_buy_count = len(buy_reasons) - strong_buy_count - weak_buy_count
    sell_count = len(sell_reasons)

    buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * weak_buy_count + w_weak_buy * other_buy_count) * 0.1
    sell_adjustment = w_sell * sell_count * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    reasons = []
    if buy_reasons:
        reasons.append("Buy Signals:")
        for reason in buy_reasons:
            reasons.append(f"  - {reason}")
    if sell_reasons:
        reasons.append("Sell Signals:")
        for reason in sell_reasons:
            reasons.append(f"  - {reason}")
    if not reasons:
        reasons.append("- No significant signals detected.")

    return target_weight, reasons

def calculate_required_shares(target_tsla_weight, target_tsll_weight, total_value, tsla_close, tsll_close):
    """추천 비중에 맞추기 위해 필요한 TSLA 및 TSLL 주식 수를 계산합니다."""
    target_tsla_value = target_tsla_weight * total_value
    target_tsll_value = target_tsll_weight * total_value
    required_tsla_shares = int(target_tsla_value / tsla_close)
    required_tsll_shares = int(target_tsll_value / tsll_close)
    return required_tsla_shares, required_tsll_shares

def adjust_portfolio(target_tsll_weight, current_tsll_weight, total_value, tsll_close):
    """포트폴리오 조정 제안을 출력합니다."""
    target_tsll_value = target_tsll_weight * total_value
    current_tsll_value = current_tsll_weight * total_value
    difference = target_tsll_value - current_tsll_value
    shares_to_adjust = int(difference / tsll_close)

    if abs(difference) < 100:
        print(" - No significant adjustment needed.")
    elif difference > 0:
        print(f" - Buy TSLL: ${difference:.2f} (approx. {shares_to_adjust} shares)")
    else:
        print(f" - Sell TSLL: ${-difference:.2f} (approx. {-shares_to_adjust} shares)")

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

    tsla_close = tsla_df['Close'].iloc[-1]
    tsll_close = tsll_df['Close'].iloc[-1]
    data_date = tsla_df.index[-1].strftime('%Y-%m-%d')

    daily_rsi = tsla_df['RSI'].iloc[-1]
    daily_rsi_trend = get_rsi_trend(tsla_df['RSI'].tail(10))
    close = tsla_df['Close'].iloc[-1]
    sma50 = tsla_df['SMA50'].iloc[-1]
    sma200 = tsla_df['SMA200'].iloc[-1]
    macd = tsla_df['MACD'].iloc[-1]
    macd_signal = tsla_df['MACD_signal'].iloc[-1]
    upper_band = tsla_df['Upper Band'].iloc[-1]
    lower_band = tsla_df['Lower Band'].iloc[-1]
    volume_change = tsla_df['Volume Change'].iloc[-1]
    atr = tsla_df['ATR'].iloc[-1]  # ATR 값 추가

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
        ["Volume Change", f"{volume_change:.2%}", "-"],
        ["ATR", f"${atr:.2f}", "-"]
    ]
    print(f"\n### Market Indicators (as of {data_date})")
    print(tabulate(indicators, headers=["Indicator", "Value", "Trend/Notes"], tablefmt="fancy_grid"))

    # 현재 주가 표시
    print("\n### Current Stock Prices")
    print(f"- **TSLA Close**: ${tsla_close:.2f}")
    print(f"- **TSLL Close**: ${tsll_close:.2f}")

    # transactions.txt 파일 유무 확인 및 처리
    current_holdings, initial_investment = load_transactions()
    if not current_holdings:
        total_value = initial_portfolio_value
        current_tsll_weight = 0.0
        print("\n### Current Portfolio")
        print(f"- Initial Portfolio Value: ${initial_portfolio_value:.2f}")
        print("- No holdings found. Assuming initial portfolio value for recommendations.")
    else:
        total_value, tsla_value, tsll_value, current_tsla_weight, current_tsll_weight, returns = calculate_portfolio_metrics(
            current_holdings, tsla_close, tsll_close, initial_investment
        )
        print("\n### Current Portfolio")
        print(f"- Initial Investment: ${initial_investment:.2f}")
        print(f"- TSLA: {current_holdings.get('TSLA', 0)} shares, value: ${tsla_value:.2f}")
        print(f"- TSLL: {current_holdings.get('TSLL', 0)} shares, value: ${tsll_value:.2f}")
        print(f"- Total Portfolio Value: ${total_value:.2f}")
        print(f"- TSLA Weight: {current_tsla_weight*100:.2f}%")
        print(f"- TSLL Weight: {current_tsll_weight*100:.2f}%")
        print(f"- Portfolio Returns: {returns:.2f}%")

    # 시장 지표를 기반으로 목표 TSLL 비중 계산
    target_tsll_weight, reasons = get_target_tsll_weight(
        fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr,
        lower_band, upper_band, current_tsll_weight, optimal_params
    )
    target_tsla_weight = 1 - target_tsll_weight

    # 필요한 주식 수 계산
    required_tsla_shares, required_tsll_shares = calculate_required_shares(
        target_tsla_weight, target_tsll_weight, total_value, tsla_close, tsll_close
    )

    # 추천 비중 및 필요한 주식 수 표시
    print("\n### Recommended Portfolio Weights")
    print(f"- **TSLA Weight**: {target_tsla_weight*100:.0f}% (approx. {required_tsla_shares} shares)")
    print(f"- **TSLL Weight**: {target_tsll_weight*100:.0f}% (approx. {required_tsll_shares} shares)")

    # 비중 조정 제안
    print("\n### Portfolio Adjustment Suggestion")
    adjust_portfolio(target_tsll_weight, current_tsll_weight, total_value, tsll_close)

    # 비중 조정 이유
    print("\n### Adjustment Reasons")
    for reason in reasons:
        print(reason)

    log_decision(data_date, target_tsll_weight, reasons)

if __name__ == "__main__":
    analyze_and_recommend()
