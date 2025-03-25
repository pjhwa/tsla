import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.stats import linregress
import json
import os
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리 추가

# 데이터 로드 및 전처리
def load_data():
    """주가 데이터와 공포탐욕지수를 로드하고 전처리합니다."""
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    # Volume 열의 쉼표 제거 및 float 변환
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 데이터 병합
    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)
    data = data.dropna()

    # 지표 계산
    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    data['Upper Band_TSLA'], _, data['Lower Band_TSLA'] = calculate_bollinger_bands(data['Close_TSLA'])
    data['MACD_TSLA'], data['MACD_signal_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']], 14)

    # 주간 RSI 계산 및 리인덱싱
    weekly_rsi = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14)
    data['Weekly RSI_TSLA'] = weekly_rsi.reindex(data.index, method='ffill').fillna(50)

    return data

# 지표 계산 함수
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
    high_low = df['High_TSLA'] - df['Low_TSLA']
    high_close = np.abs(df['High_TSLA'] - df['Close_TSLA'].shift())
    low_close = np.abs(df['Low_TSLA'] - df['Close_TSLA'].shift())
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

# 백테스트 및 적합도 함수 (tqdm 추가)
def evaluate(individual, data):
    """주어진 파라미터로 백테스트를 수행하고 수익률을 계산합니다."""
    params = {
        "fg_buy": individual[0],
        "fg_sell": individual[1],
        "daily_rsi_buy": individual[2],
        "daily_rsi_sell": individual[3],
        "weekly_rsi_buy": individual[4],
        "weekly_rsi_sell": individual[5],
        "volume_change_strong_buy": individual[6],
        "volume_change_weak_buy": individual[7],
        "volume_change_sell": individual[8],
        "w_strong_buy": individual[9],
        "w_weak_buy": individual[10],
        "w_sell": individual[11]
    }

    portfolio_value = 100000  # 초기 자산 $100,000
    holdings = {'TSLA': 0, 'TSLL': 0}
    current_tsll_weight = 0.0

    # tqdm으로 백테스트 진행 상황 표시
    for i in tqdm(range(1, len(data)), desc="Backtesting", leave=False):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]

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

        # 목표 비중 계산
        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr,
            lower_band, upper_band, current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        # 포트폴리오 조정
        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + portfolio_value
        required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])
        required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])

        # 매수/매도
        if required_tsla_shares > holdings['TSLA']:
            buy_shares = required_tsla_shares - holdings['TSLA']
            cost = buy_shares * row['Close_TSLA']
            if portfolio_value >= cost:
                portfolio_value -= cost
                holdings['TSLA'] += buy_shares
        elif required_tsla_shares < holdings['TSLA']:
            sell_shares = holdings['TSLA'] - required_tsla_shares
            proceeds = sell_shares * row['Close_TSLA']
            portfolio_value += proceeds
            holdings['TSLA'] -= sell_shares

        if required_tsll_shares > holdings['TSLL']:
            buy_shares = required_tsll_shares - holdings['TSLL']
            cost = buy_shares * row['Close_TSLL']
            if portfolio_value >= cost:
                portfolio_value -= cost
                holdings['TSLL'] += buy_shares
        elif required_tsll_shares < holdings['TSLL']:
            sell_shares = holdings['TSLL'] - required_tsll_shares
            proceeds = sell_shares * row['Close_TSLL']
            portfolio_value += proceeds
            holdings['TSLL'] -= sell_shares

        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0

    final_value = holdings['TSLA'] * data['Close_TSLA'].iloc[-1] + holdings['TSLL'] * data['Close_TSLL'].iloc[-1] + portfolio_value
    return (final_value - 100000) / 100000,  # 수익률 반환

# 비중 계산 로직
def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr, lower_band, upper_band, current_tsll_weight, params):
    """TSLL의 목표 비중을 계산합니다."""
    base_weight = current_tsll_weight

    # 동적 임계값 계산 (ATR 기반)
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

# 유전 알고리즘 설정
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
param_ranges = [
    (10, 40),  # fg_buy
    (60, 90),  # fg_sell
    (20, 40),  # daily_rsi_buy
    (60, 80),  # daily_rsi_sell
    (20, 40),  # weekly_rsi_buy
    (60, 80),  # weekly_rsi_sell
    (0.3, 0.7),  # volume_change_strong_buy
    (0.1, 0.3),  # volume_change_weak_buy
    (-0.3, -0.1),  # volume_change_sell
    (1.5, 2.5),  # w_strong_buy
    (0.5, 1.5),  # w_weak_buy
    (0.5, 1.5)   # w_sell
]

for i, (low, high) in enumerate(param_ranges):
    toolbox.register(f"attr_{i}", random.uniform, low, high)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                  toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7,
                  toolbox.attr_8, toolbox.attr_9, toolbox.attr_10, toolbox.attr_11], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, data=load_data())
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 메인 실행 함수 (tqdm 추가)
def main():
    """유전 알고리즘을 실행하여 최적 파라미터를 찾습니다."""
    population = toolbox.population(n=100)
    ngen = 50  # 세대 수

    # tqdm으로 세대별 진행 상황 표시
    for gen in tqdm(range(ngen), desc="Genetic Algorithm Progress"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    # 최적 결과 출력
    best_ind = tools.selBest(population, 1)[0]
    best_params = {
        "fg_buy": best_ind[0],
        "fg_sell": best_ind[1],
        "daily_rsi_buy": best_ind[2],
        "daily_rsi_sell": best_ind[3],
        "weekly_rsi_buy": best_ind[4],
        "weekly_rsi_sell": best_ind[5],
        "volume_change_strong_buy": best_ind[6],
        "volume_change_weak_buy": best_ind[7],
        "volume_change_sell": best_ind[8],
        "w_strong_buy": best_ind[9],
        "w_weak_buy": best_ind[10],
        "w_sell": best_ind[11]
    }
    print("최적 파라미터:", best_params)
    print("최대 수익률:", evaluate(best_ind, load_data())[0])

    # 최적 파라미터 저장
    with open("optimal_params.json", "w") as f:
        json.dump(best_params, f)

if __name__ == "__main__":
    main()
