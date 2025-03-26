import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.stats import linregress
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 전역 변수 초기화
data = None
volatility = None

def init_process():
    """각 프로세스에서 데이터를 로드하도록 초기화 함수"""
    global data, volatility
    fear_greed_df = pd.read_csv('fear_greed_2years.csv')
    tsla_df = pd.read_csv('TSLA-history-2y.csv')
    tsll_df = pd.read_csv('TSLL-history-2y.csv')

    fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], errors='coerce')
    tsla_df['Date'] = pd.to_datetime(tsla_df['Date'], errors='coerce')
    tsll_df['Date'] = pd.to_datetime(tsll_df['Date'], errors='coerce')

    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.interpolate(method='linear')

    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    data['Upper Band_TSLA'], _, data['Lower Band_TSLA'] = calculate_bollinger_bands(data['Close_TSLA'])
    data['MACD_TSLA'], data['MACD_signal_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']], 14)

    weekly_rsi = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14)
    data['Weekly RSI_TSLA'] = weekly_rsi.reindex(data.index, method='ffill').fillna(50)

    volatility = get_volatility_for_backtest(data)

def get_volatility_for_backtest(data):
    """VIX 지수를 로드하고 평균 변동성을 계산"""
    vix_df = pd.read_csv('VIX-history-2y.csv', skiprows=3, header=None, names=['Date', 'VIX'],
                         parse_dates=['Date'], index_col='Date')
    start_date = data.index.min()
    end_date = data.index.max()
    vix_filtered = vix_df.loc[start_date:end_date]
    vix_filtered['VIX'] = vix_filtered['VIX'].interpolate(method='linear')
    return vix_filtered['VIX'].mean()

def get_dynamic_param_ranges(volatility):
    """변동성에 따른 동적 파라미터 범위 설정"""
    if volatility > 30:  # 고변동성
        return [
            (15, 50), (50, 90), (20, 50), (60, 90), (25, 50), (60, 90),  # fg_buy, fg_sell, daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell
            (0.2, 1.0), (0.05, 0.5), (-1.0, -0.05), (1.0, 3.0), (0.5, 2.0), (0.5, 2.0)  # volume_change_strong_buy, volume_change_weak_buy, volume_change_sell, w_strong_buy, w_weak_buy, w_sell
        ]
    return [  # 저변동성
        (20, 40), (60, 85), (25, 40), (65, 85), (30, 40), (65, 85),  # fg_buy, fg_sell, daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell
        (0.3, 1.0), (0.1, 0.5), (-1.0, -0.1), (1.5, 3.0), (0.5, 2.0), (0.5, 2.0)  # volume_change_strong_buy, volume_change_weak_buy, volume_change_sell, w_strong_buy, w_weak_buy, w_sell
    ]

def calculate_rsi(series, timeperiod=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).interpolate(method='linear')

def calculate_sma(series, timeperiod=20):
    return series.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd.interpolate(method='linear'), macd_signal.interpolate(method='linear')

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper = sma + (std * nbdevup)
    lower = sma - (std * nbdevdn)
    return upper.interpolate(method='linear'), sma.interpolate(method='linear'), lower.interpolate(method='linear')

def calculate_atr(df, timeperiod=14):
    high_low = df['High_TSLA'] - df['Low_TSLA']
    high_close = np.abs(df['High_TSLA'] - df['Close_TSLA'].shift())
    low_close = np.abs(df['Low_TSLA'] - df['Close_TSLA'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean().interpolate(method='linear')

def get_rsi_trend(rsi_series, window=10):
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    if slope > 0.1:
        return "Increasing"
    elif slope < -0.1:
        return "Decreasing"
    return "Stable"

def evaluate(individual):
    global data, volatility
    params = {
        "fg_buy": individual[0], "fg_sell": individual[1], "daily_rsi_buy": individual[2],
        "daily_rsi_sell": individual[3], "weekly_rsi_buy": individual[4], "weekly_rsi_sell": individual[5],
        "volume_change_strong_buy": individual[6], "volume_change_weak_buy": individual[7],
        "volume_change_sell": individual[8], "w_strong_buy": individual[9], "w_weak_buy": individual[10],
        "w_sell": individual[11]
    }

    portfolio_value = 100000
    holdings = {'TSLA': 0, 'TSLL': 0}
    current_tsll_weight = 0.0
    portfolio_values = []

    for i in range(1, len(data)):
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

        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal,
            volume_change, atr, lower_band, upper_band, current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + portfolio_value
        portfolio_values.append(total_value)

        required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])
        required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])

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
    portfolio_values.append(final_value)

    # 수익률 및 Sharpe Ratio 계산
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = (final_value - 100000) / 100000  # 총 수익률
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0

    # 수익률에 가중치를 둔 Sharpe Ratio 반환
    weighted_fitness = sharpe_ratio * (1 + total_return)

    return weighted_fitness,

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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def clip_individual(ind, param_ranges):
    """파라미터를 지정된 범위 내로 클리핑"""
    for i in range(len(ind)):
        low, high = param_ranges[i]
        ind[i] = max(low, min(ind[i], high))
    return ind

def main():
    global volatility
    init_process()  # 메인 프로세스에서 초기 데이터 로드
    param_ranges = get_dynamic_param_ranges(volatility)

    # 개별 파라미터 생성 함수 등록
    for i, (low, high) in enumerate(param_ranges):
        toolbox.register(f"attr_{i}", random.uniform, low, high)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                      toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7,
                      toolbox.attr_8, toolbox.attr_9, toolbox.attr_10, toolbox.attr_11], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    processes = min(cpu_count(), 4)
    pool = Pool(processes=processes, initializer=init_process)
    toolbox.register("map", pool.map)

    population = toolbox.population(n=100)
    ngen = 50
    patience = 10
    best_fitness = -np.inf
    patience_counter = 0

    for gen in tqdm(range(ngen), desc="Genetic Algorithm Progress"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)

        # 돌연변이와 교차 후 클리핑 적용
        for ind in offspring:
            clip_individual(ind, param_ranges)

        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

        current_best = max(population, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
        if current_best > best_fitness:
            best_fitness = current_best
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"{gen}세대에서 조기 종료: {patience}세대 동안 개선 없음")
            break

    best_ind = tools.selBest(population, 1)[0]
    best_params = {
        "fg_buy": best_ind[0], "fg_sell": best_ind[1], "daily_rsi_buy": best_ind[2],
        "daily_rsi_sell": best_ind[3], "weekly_rsi_buy": best_ind[4], "weekly_rsi_sell": best_ind[5],
        "volume_change_strong_buy": best_ind[6], "volume_change_weak_buy": best_ind[7],
        "volume_change_sell": best_ind[8], "w_strong_buy": best_ind[9], "w_weak_buy": best_ind[10],
        "w_sell": best_ind[11]
    }
    print("최적 파라미터:", best_params)
    print("최대 가중치 Sharpe Ratio:", evaluate(best_ind)[0])

    with open("optimal_params.json", "w") as f:
        json.dump(best_params, f)

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
