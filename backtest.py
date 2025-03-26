import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.stats import linregress
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 상수 정의
POPULATION_SIZE = 200
NUM_GENERATIONS = 100
PATIENCE = 15

# 전역 변수 초기화
data = None
volatility = None

def load_and_preprocess_data(file_path, date_column='Date', volume_column='Volume'):
    """CSV 파일을 로드하고 전처리합니다."""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if volume_column in df.columns:
        df[volume_column] = df[volume_column].str.replace(',', '').astype(float)
    return df

def merge_dataframes(tsla_df, tsll_df, fear_greed_df):
    """데이터프레임을 'Date' 열을 기준으로 병합합니다."""
    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.interpolate(method='linear')
    return data

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
    """MACD와 신호선을 계산합니다."""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd.interpolate(method='linear'), macd_signal.interpolate(method='linear')

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """볼린저 밴드를 계산합니다."""
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper = sma + (std * nbdevup)
    lower = sma - (std * nbdevdn)
    return upper.interpolate(method='linear'), sma.interpolate(method='linear'), lower.interpolate(method='linear')

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

def init_process():
    """데이터를 초기화하고 지표를 계산합니다."""
    global data, volatility
    fear_greed_df = load_and_preprocess_data('fear_greed_2years.csv', date_column='date')
    tsla_df = load_and_preprocess_data('TSLA-history-2y.csv')
    tsll_df = load_and_preprocess_data('TSLL-history-2y.csv')

    data = merge_dataframes(tsla_df, tsll_df, fear_greed_df)

    # 지표 계산
    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    upper_band, _, lower_band = calculate_bollinger_bands(data['Close_TSLA'])
    data['Upper Band_TSLA'] = upper_band
    data['Lower Band_TSLA'] = lower_band
    data['MACD_TSLA'], data['MACD_signal_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']], 14)
    data['Stochastic_K_TSLA'], data['Stochastic_D_TSLA'] = calculate_stochastic(data[['High_TSLA', 'Low_TSLA', 'Close_TSLA']])
    data['OBV_TSLA'] = calculate_obv(data['Close_TSLA'], data['Volume_TSLA'])
    data['BB_width_TSLA'] = (data['Upper Band_TSLA'] - data['Lower Band_TSLA']) / data['SMA50_TSLA']
    data['MACD_histogram_TSLA'] = data['MACD_TSLA'] - data['MACD_signal_TSLA']

    weekly_rsi = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14)
    data['Weekly RSI_TSLA'] = weekly_rsi.reindex(data.index, method='ffill').fillna(50)

    volatility = get_volatility_for_backtest(data)

def get_volatility_for_backtest(data):
    """VIX 지수를 로드하고 평균 변동성을 계산합니다."""
    vix_df = pd.read_csv('VIX-history-2y.csv', skiprows=3, header=None, names=['Date', 'VIX'],
                         parse_dates=['Date'], index_col='Date')
    start_date = data.index.min()
    end_date = data.index.max()
    vix_filtered = vix_df.loc[start_date:end_date]
    vix_filtered['VIX'] = vix_filtered['VIX'].interpolate(method='linear')
    return vix_filtered['VIX'].mean()

def get_dynamic_param_ranges(volatility):
    """변동성에 따라 동적 파라미터 범위를 설정합니다."""
    if volatility > 30:  # 고변동성
        return [
            (15, 50), (50, 90), (20, 50), (60, 90), (25, 50), (60, 90),  # fg_buy, fg_sell, daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell
            (0.2, 1.0), (0.05, 0.5), (-1.0, -0.05),  # volume_change_strong_buy, volume_change_weak_buy, volume_change_sell
            (1.0, 3.0), (0.5, 2.0), (0.5, 2.0),  # w_strong_buy, w_weak_buy, w_sell
            (20, 40), (60, 80),  # stochastic_buy, stochastic_sell
            (0.5, 2.0), (0.5, 2.0)  # obv_weight, bb_width_weight
        ]
    return [  # 저변동성
        (20, 40), (60, 85), (25, 40), (65, 85), (30, 40), (65, 85),
        (0.3, 1.0), (0.1, 0.5), (-1.0, -0.1),
        (1.5, 3.0), (0.5, 2.0), (0.5, 2.0),
        (20, 40), (60, 80),
        (0.5, 2.0), (0.5, 2.0)
    ]

def get_rsi_trend(rsi_series, window=10):
    """RSI 추세를 분석합니다."""
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    return "Increasing" if slope > 0.1 else "Decreasing" if slope < -0.1 else "Stable"

def adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row):
    """포트폴리오를 목표 비중에 맞춰 조정합니다."""
    total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
    required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])
    required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])

    # TSLA 조정
    if required_tsla_shares > holdings['TSLA']:
        buy_shares = required_tsla_shares - holdings['TSLA']
        cost = buy_shares * row['Close_TSLA']
        if cash >= cost:
            cash -= cost
            holdings['TSLA'] += buy_shares
    elif required_tsla_shares < holdings['TSLA']:
        sell_shares = holdings['TSLA'] - required_tsla_shares
        proceeds = sell_shares * row['Close_TSLA']
        cash += proceeds
        holdings['TSLA'] -= sell_shares

    # TSLL 조정
    if required_tsll_shares > holdings['TSLL']:
        buy_shares = required_tsll_shares - holdings['TSLL']
        cost = buy_shares * row['Close_TSLL']
        if cash >= cost:
            cash -= cost
            holdings['TSLL'] += buy_shares
    elif required_tsll_shares < holdings['TSLL']:
        sell_shares = holdings['TSLL'] - required_tsll_shares
        proceeds = sell_shares * row['Close_TSLL']
        cash += proceeds
        holdings['TSLL'] -= sell_shares

    return holdings, cash

def calculate_fitness(portfolio_values):
    """포트폴리오 성과를 기반으로 피트니스 스코어를 계산합니다."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    return sharpe_ratio / (1 + max_drawdown)  # Calmar Ratio 기반

def evaluate(individual, data_subset=None):
    """롤링 윈도우 백테스팅을 위한 평가 함수입니다."""
    global data
    if data_subset is None:
        data_subset = data

    params = {
        "fg_buy": individual[0], "fg_sell": individual[1], "daily_rsi_buy": individual[2],
        "daily_rsi_sell": individual[3], "weekly_rsi_buy": individual[4], "weekly_rsi_sell": individual[5],
        "volume_change_strong_buy": individual[6], "volume_change_weak_buy": individual[7],
        "volume_change_sell": individual[8], "w_strong_buy": individual[9], "w_weak_buy": individual[10],
        "w_sell": individual[11], "stochastic_buy": individual[12], "stochastic_sell": individual[13],
        "obv_weight": individual[14], "bb_width_weight": individual[15]
    }

    cash = 100000  # 초기 현금
    holdings = {'TSLA': 0, 'TSLL': 0}
    current_tsll_weight = 0.0
    portfolio_values = []

    for i in range(1, len(data_subset)):
        row = data_subset.iloc[i]
        fear_greed = row['y']
        daily_rsi = row['RSI_TSLA']
        weekly_rsi = row['Weekly RSI_TSLA']
        daily_rsi_trend = get_rsi_trend(data_subset['RSI_TSLA'].iloc[max(0, i-10):i+1])
        close = row['Close_TSLA']
        sma50 = row['SMA50_TSLA']
        sma200 = row['SMA200_TSLA']
        macd = row['MACD_TSLA']
        macd_signal = row['MACD_signal_TSLA']
        volume_change = row['Volume Change_TSLA'] if not pd.isna(row['Volume Change_TSLA']) else 0
        atr = row['ATR_TSLA']
        lower_band = row['Lower Band_TSLA']
        upper_band = row['Upper Band_TSLA']
        stochastic_k = row['Stochastic_K_TSLA']
        stochastic_d = row['Stochastic_D_TSLA']
        obv = row['OBV_TSLA']
        obv_prev = data_subset['OBV_TSLA'].iloc[i-1] if i > 0 else obv
        bb_width = row['BB_width_TSLA']
        macd_histogram = row['MACD_histogram_TSLA']

        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal,
            volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, macd_histogram, current_tsll_weight, params
        )
        target_tsla_weight = 1 - target_tsll_weight

        holdings, cash = adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)
        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
        portfolio_values.append(total_value)
        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0

    final_value = holdings['TSLA'] * data_subset['Close_TSLA'].iloc[-1] + holdings['TSLL'] * data_subset['Close_TSLL'].iloc[-1] + cash
    portfolio_values.append(final_value)

    fitness = calculate_fitness(portfolio_values)
    return fitness,

def calculate_max_drawdown(portfolio_values):
    """최대 손실(Maximum Drawdown)을 계산합니다."""
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def get_target_tsll_weight(fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma50, sma200, macd, macd_signal, volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, macd_histogram, current_tsll_weight, params):
    """TSLL의 목표 비중을 계산합니다."""
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
        (daily_rsi_trend == "Increasing") and (close > sma200),
        stochastic_k < params["stochastic_buy"],
        obv > obv_prev,
        bb_width < 0.05,
        macd_histogram > 0
    ]

    sell_conditions = [
        fear_greed >= params["fg_sell"],
        daily_rsi > params["daily_rsi_sell"],
        (macd < macd_signal) and (macd_signal > 0),
        volume_change < volume_change_sell,
        close > upper_band,
        (daily_rsi_trend == "Decreasing") and (close < sma200),
        stochastic_k > params["stochastic_sell"],
        obv < obv_prev,
        bb_width > 0.15,
        macd_histogram < 0
    ]

    strong_buy_count = buy_conditions[3]
    weak_buy_count = buy_conditions[4] and not strong_buy_count
    other_buy_count = sum(buy_conditions[:3] + buy_conditions[5:10])
    sell_count = sum(sell_conditions[:6])
    obv_buy = buy_conditions[8]
    obv_sell = sell_conditions[7]
    bb_width_buy = buy_conditions[9]
    bb_width_sell = sell_conditions[8]
    macd_hist_buy = buy_conditions[10]
    macd_hist_sell = sell_conditions[9]

    w_strong_buy = params["w_strong_buy"]
    w_weak_buy = params["w_weak_buy"]
    w_sell = params["w_sell"]
    obv_weight = params["obv_weight"]
    bb_width_weight = params["bb_width_weight"]

    buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * weak_buy_count + w_weak_buy * other_buy_count + obv_weight * obv_buy + bb_width_weight * bb_width_buy + w_weak_buy * macd_hist_buy) * 0.1
    sell_adjustment = (w_sell * sell_count + obv_weight * obv_sell + bb_width_weight * bb_width_sell + w_sell * macd_hist_sell) * 0.1
    target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))

    return target_weight, []

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def clip_individual(ind, param_ranges):
    """파라미터를 지정된 범위 내로 클리핑합니다."""
    for i in range(len(ind)):
        low, high = param_ranges[i]
        ind[i] = max(low, min(ind[i], high))
    return ind

def cross_validation_evaluation(individual, folds=5):
    """교차 검증을 통해 평가합니다."""
    global data
    fold_size = len(data) // folds
    fitness_scores = []

    for i in range(folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < folds - 1 else len(data)
        test_data = data.iloc[start_idx:end_idx]
        fitness = evaluate(individual, test_data)[0]
        fitness_scores.append(fitness)

    return np.mean(fitness_scores),

def setup_toolbox(param_ranges):
    """DEAP 툴박스를 설정합니다."""
    for i, (low, high) in enumerate(param_ranges):
        toolbox.register(f"attr_{i}", random.uniform, low, high)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                      toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7,
                      toolbox.attr_8, toolbox.attr_9, toolbox.attr_10, toolbox.attr_11,
                      toolbox.attr_12, toolbox.attr_13, toolbox.attr_14, toolbox.attr_15], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", cross_validation_evaluation)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

def save_best_params(best_params):
    """최적 파라미터를 JSON 파일로 저장합니다."""
    data_to_save = {
        "version": "2.0",
        "parameters": best_params
    }
    with open("optimal_params.json", "w") as f:
        json.dump(data_to_save, f)

def main():
    """메인 함수: 유전 알고리즘을 실행하고 최적 파라미터를 찾습니다."""
    global volatility
    init_process()
    param_ranges = get_dynamic_param_ranges(volatility)
    setup_toolbox(param_ranges)

    processes = min(cpu_count(), 4)
    pool = Pool(processes=processes, initializer=init_process)
    toolbox.register("map", pool.map)

    population = toolbox.population(n=POPULATION_SIZE)
    best_fitness = -np.inf
    patience_counter = 0

    for gen in tqdm(range(NUM_GENERATIONS), desc="유전 알고리즘 진행"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.4)
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

        if patience_counter >= PATIENCE:
            print(f"{gen}세대에서 조기 종료: {PATIENCE}세대 동안 개선 없음")
            break

    best_ind = tools.selBest(population, 1)[0]
    best_params = {
        "fg_buy": best_ind[0], "fg_sell": best_ind[1], "daily_rsi_buy": best_ind[2],
        "daily_rsi_sell": best_ind[3], "weekly_rsi_buy": best_ind[4], "weekly_rsi_sell": best_ind[5],
        "volume_change_strong_buy": best_ind[6], "volume_change_weak_buy": best_ind[7],
        "volume_change_sell": best_ind[8], "w_strong_buy": best_ind[9], "w_weak_buy": best_ind[10],
        "w_sell": best_ind[11], "stochastic_buy": best_ind[12], "stochastic_sell": best_ind[13],
        "obv_weight": best_ind[14], "bb_width_weight": best_ind[15]
    }
    print("최적 파라미터:", best_params)
    print("최대 피트니스 (Calmar Ratio 기반):", evaluate(best_ind)[0])

    save_best_params(best_params)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
