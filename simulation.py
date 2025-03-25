import pandas as pd
import numpy as np
import json
import argparse
import datetime
from tabulate import tabulate
from dateutil.relativedelta import relativedelta

# 초기 설정
initial_portfolio_value = 100000  # 초기 자산 $100,000
default_end_date = pd.Timestamp('2025-03-25')  # 기본 끝 날짜

### 최적 파라미터 로드 함수 ###
def load_optimal_params(file_path="optimal_params.json"):
    """JSON 파일에서 최적 파라미터를 로드합니다. 파일이 없으면 기본값을 반환합니다."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"최적 파라미터 파일({file_path})이 없습니다. 기본값을 사용합니다.")
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

### 데이터 로드 및 전처리 ###
def load_and_prepare_data(period, start_date=None):
    """CSV 파일에서 데이터를 로드하고 시뮬레이션 기간에 맞게 필터링합니다."""
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    # Volume 데이터 전처리
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 인덱스 설정
    fear_greed_df.set_index('date', inplace=True)
    tsla_df.set_index('Date', inplace=True)
    tsll_df.set_index('Date', inplace=True)

    # 데이터 병합
    data = pd.concat([tsla_df, tsll_df.add_prefix('TSLL_'), fear_greed_df['y']], axis=1).dropna()

    # 기술적 지표 계산
    data['Daily RSI'] = calculate_rsi(data['Close'], timeperiod=14)
    data['Weekly RSI'] = calculate_rsi(data['Close'].resample('W').last(), timeperiod=14).reindex(data.index, method='ffill')
    data['SMA50'] = calculate_sma(data['Close'], timeperiod=50)
    data['SMA200'] = calculate_sma(data['Close'], timeperiod=200)
    data['Upper Band'], data['Middle Band'], data['Lower Band'] = calculate_bollinger_bands(data['Close'])
    data['MACD'], data['MACD_signal'] = calculate_macd(data['Close'])
    data['Volume Change'] = data['Volume'].pct_change().fillna(0)
    data['ATR'] = calculate_atr(data, timeperiod=14)

    # 시뮬레이션 기간 설정
    if start_date:
        start_date = pd.Timestamp(start_date)
        if period == '1M':
            end_date = start_date + relativedelta(months=1)
        elif period == '3M':
            end_date = start_date + relativedelta(months=3)
        elif period == '6M':
            end_date = start_date + relativedelta(months=6)
        elif period == '1Y':
            end_date = start_date + relativedelta(years=1)
        else:
            raise ValueError("Invalid period. Use '1M', '3M', '6M', or '1Y'.")
    else:
        end_date = default_end_date
        if period == '1M':
            start_date = end_date - pd.offsets.MonthEnd(1)
        elif period == '3M':
            start_date = end_date - pd.offsets.MonthEnd(3)
        elif period == '6M':
            start_date = end_date - pd.offsets.MonthEnd(6)
        elif period == '1Y':
            start_date = end_date - pd.offsets.DateOffset(years=1)
        else:
            raise ValueError("Invalid period. Use '1M', '3M', '6M', or '1Y'.")

    # 기간 필터링
    data = data.loc[start_date:end_date]
    return data, start_date, end_date

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

### 비중 계산 함수 ###
def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, close, sma50, sma200, macd, macd_signal, volume_change, atr, lower_band, upper_band, current_tsll_weight, params):
    """TSLL의 목표 비중을 계산하고 조정 이유를 반환합니다."""
    base_weight = current_tsll_weight
    reasons = []

    # 동적 임계값 계산 (ATR 기반)
    atr_normalized = atr / close
    volume_change_strong_buy = params["volume_change_strong_buy"] * (1 + atr_normalized)
    volume_change_weak_buy = params["volume_change_weak_buy"] * (1 + atr_normalized)
    volume_change_sell = params["volume_change_sell"] * (1 + atr_normalized)

    buy_conditions = {
        f"Fear & Greed Index ≤ {params['fg_buy']}": fear_greed <= params["fg_buy"],
        f"Daily RSI < {params['daily_rsi_buy']}": daily_rsi < params["daily_rsi_buy"],
        "MACD > MACD Signal and MACD Signal < 0": (macd > macd_signal) and (macd_signal < 0),
        f"Volume Change > {volume_change_strong_buy:.2f} (Strong Buy)": volume_change > volume_change_strong_buy,
        f"Volume Change > {volume_change_weak_buy:.2f} (Weak Buy)": volume_change > volume_change_weak_buy,
        "Close < Lower Bollinger Band": close < lower_band
    }

    sell_conditions = {
        f"Fear & Greed Index ≥ {params['fg_sell']}": fear_greed >= params["fg_sell"],
        f"Daily RSI > {params['daily_rsi_sell']}": daily_rsi > params["daily_rsi_sell"],
        "MACD < MACD Signal and MACD Signal > 0": (macd < macd_signal) and (macd_signal > 0),
        f"Volume Change < {volume_change_sell:.2f}": volume_change < volume_change_sell,
        "Close > Upper Bollinger Band": close > upper_band
    }

    buy_reasons = [condition for condition, is_true in buy_conditions.items() if is_true]
    sell_reasons = [condition for condition, is_true in sell_conditions.items() if is_true]

    # 세분화된 가중치 적용
    w_strong_buy = params["w_strong_buy"]
    w_weak_buy = params["w_weak_buy"]
    w_sell = params["w_sell"]

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

### 시뮬레이션 로직 ###
def simulate_portfolio(data, optimal_params):
    """주어진 데이터를 기반으로 포트폴리오 시뮬레이션을 실행합니다."""
    portfolio_value = np.ones(len(data)) * initial_portfolio_value
    current_tsll_weight = np.zeros(len(data))
    tsla_shares = np.zeros(len(data))
    tsll_shares = np.zeros(len(data))

    for i in range(1, len(data)):
        # 현재 데이터
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        fear_greed = row['y']
        daily_rsi = row['Daily RSI']
        weekly_rsi = row['Weekly RSI']
        close = row['Close']
        sma50 = row['SMA50']
        sma200 = row['SMA200']
        macd = row['MACD']
        macd_signal = row['MACD_signal']
        volume_change = row['Volume Change']
        lower_band = row['Lower Band']
        upper_band = row['Upper Band']
        atr = row['ATR']
        tsla_close = row['Close']
        tsll_close = row['TSLL_Close']

        # 목표 TSLL 비중 계산
        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, close, sma50, sma200, macd, macd_signal,
            volume_change, atr, lower_band, upper_band, current_tsll_weight[i-1], optimal_params
        )
        target_tsla_weight = 1 - target_tsll_weight
        current_tsll_weight[i] = target_tsll_weight

        # 포트폴리오 가치 계산 및 주식 수 조정
        if i == 1:  # 초기 투자
            tsla_shares[i] = int((target_tsla_weight * initial_portfolio_value) / tsla_close)
            tsll_shares[i] = int((target_tsll_weight * initial_portfolio_value) / tsll_close)
        else:
            # 이전 날짜의 주식 수 유지
            tsla_shares[i] = tsla_shares[i-1]
            tsll_shares[i] = tsll_shares[i-1]
            prev_portfolio_value = portfolio_value[i-1]
            target_tsll_value = target_tsll_weight * prev_portfolio_value
            current_tsll_value = tsll_shares[i] * tsll_close
            difference = target_tsll_value - current_tsll_value

            if abs(difference) > 100:  # $100 이상 차이 시 조정
                if difference > 0:
                    tsll_shares[i] += int(difference / tsll_close)
                else:
                    tsll_shares[i] = max(0, tsll_shares[i] + int(difference / tsll_close))
                tsla_shares[i] = int((prev_portfolio_value - (tsll_shares[i] * tsll_close)) / tsla_close)

        # 현재 포트폴리오 가치 계산
        portfolio_value[i] = tsla_shares[i] * tsla_close + tsll_shares[i] * tsll_close

    # 결과 데이터프레임 생성
    result_df = pd.DataFrame({
        'Date': data.index,
        'Portfolio Value': portfolio_value,
        'TSLL Weight': current_tsll_weight,
        'TSLA Shares': tsla_shares,
        'TSLL Shares': tsll_shares
    })
    return result_df

### 결과 출력 ###
def print_simulation_results(result_df, period, start_date, end_date):
    """시뮬레이션 결과를 출력합니다."""
    initial_value = initial_portfolio_value
    final_value = result_df['Portfolio Value'].iloc[-1]
    returns = ((final_value - initial_value) / initial_value) * 100

    print(f"\n### Portfolio Simulation Results ({period})")
    print(f"- Simulation Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"- Initial Portfolio Value: ${initial_value:.2f}")
    print(f"- Final Portfolio Value: ${final_value:.2f}")
    print(f"- Portfolio Returns: {returns:.2f}%")

    # 주요 지표 테이블
    summary = [
        ["Date", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
        ["Portfolio Value", f"${initial_value:.2f}", f"${final_value:.2f}"],
        ["TSLL Weight", f"{result_df['TSLL Weight'].iloc[0]*100:.2f}%", f"{result_df['TSLL Weight'].iloc[-1]*100:.2f}%"],
        ["TSLA Shares", int(result_df['TSLA Shares'].iloc[0]), int(result_df['TSLA Shares'].iloc[-1])],
        ["TSLL Shares", int(result_df['TSLL Shares'].iloc[0]), int(result_df['TSLL Shares'].iloc[-1])]
    ]
    print("\n### Summary Table")
    print(tabulate(summary, headers=["Metric", "Start", "End"], tablefmt="fancy_grid"))

### 메인 함수 ###
def main():
    parser = argparse.ArgumentParser(description="Portfolio Simulation")
    parser.add_argument('--period', type=str, choices=['1M', '3M', '6M', '1Y'], required=True, help="Simulation period: 1M, 3M, 6M, or 1Y")
    parser.add_argument('--start_date', type=str, help="Start date for simulation (YYYY-MM-DD). If not provided, uses default end date.")
    parser.add_argument('--opt', type=str, default="optimal_params.json", help="Path to optimal parameters JSON file")
    args = parser.parse_args()

    # 최적 파라미터 로드
    optimal_params = load_optimal_params(args.opt)
    print(f"Using optimal parameters from {args.opt}: {optimal_params}")

    # 데이터 로드 및 준비
    print("데이터 로드 및 전처리 중...")
    data, start_date, end_date = load_and_prepare_data(args.period, args.start_date)
    if data.empty:
        print("선택한 기간에 데이터가 없습니다. 프로그램을 종료합니다.")
        return

    # 시뮬레이션 실행
    print(f"\n{args.period} 기간에 대한 포트폴리오 시뮬레이션 시작...")
    result_df = simulate_portfolio(data, optimal_params)

    # 결과 출력
    print_simulation_results(result_df, args.period, start_date, end_date)

    # 로그 저장
    log_filename = f"simulation_log_{args.period}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    result_df.to_csv(log_filename, index=False)
    print(f"시뮬레이션 로그가 '{log_filename}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
