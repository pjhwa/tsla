import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.stats import linregress
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
from sklearn.model_selection import TimeSeriesSplit
import logging

# 부모 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weight_adjustment import get_target_tsll_weight

# 상수 정의
POPULATION_SIZE = 400  # 인구 크기 증가로 탐색 범위 확대
NUM_GENERATIONS = 200  # 세대 수 증가로 더 깊은 최적화
PATIENCE = 25  # 조기 종료 인내심 증가
TRANSACTION_COST = 0.002  # 거래 비용 0.2% (슬리피지 포함)

# 전역 변수
data = None
volatility = None

# 로깅 설정
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path, date_column='Date', volume_column='Volume', header=0, names=None):
    """CSV 파일 로드 및 전처리"""
    if not os.path.exists(file_path):
        error_message = f"Error: '{file_path}' file not found. Please run 'collect_market_data.py' to generate the required data files."
        print(error_message)
        logging.error(error_message)
        raise FileNotFoundError(error_message)
    
    df = pd.read_csv(file_path, header=header, names=names)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if volume_column and volume_column in df.columns:
        df[volume_column] = df[volume_column].str.replace(',', '').astype(float)
    return df

def merge_dataframes(tsla_df, tsll_df, fear_greed_df):
    """데이터프레임 병합"""
    data = pd.merge(tsla_df, tsll_df, on='Date', suffixes=('_TSLA', '_TSLL'))
    data = pd.merge(data, fear_greed_df, left_on='Date', right_on='date')
    data.set_index('Date', inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.interpolate(method='linear')
    return data

def calculate_rsi(series, timeperiod=14):
    """RSI 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).interpolate(method='linear')

def calculate_sma(series, timeperiod=20):
    """SMA 계산"""
    return series.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD 계산"""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd.interpolate(method='linear'), macd_signal.interpolate(method='linear'), macd_histogram.interpolate(method='linear')

def calculate_bollinger_bands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """볼린저 밴드 계산"""
    sma = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper = sma + (std * nbdevup)
    lower = sma - (std * nbdevdn)
    return upper.interpolate(method='linear'), sma.interpolate(method='linear'), lower.interpolate(method='linear')

def calculate_atr(df, timeperiod=14):
    """ATR 계산"""
    high_low = df['High_TSLA'] - df['Low_TSLA']
    high_close = np.abs(df['High_TSLA'] - df['Close_TSLA'].shift())
    low_close = np.abs(df['Low_TSLA'] - df['Close_TSLA'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean().interpolate(method='linear')

def calculate_stochastic(df, k_period=14, d_period=3):
    """스토캐스틱 오실레이터 계산"""
    low_min = df['Low_TSLA'].rolling(window=k_period).min()
    high_max = df['High_TSLA'].rolling(window=k_period).max()
    k = 100 * ((df['Close_TSLA'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k.interpolate(method='linear'), d.interpolate(method='linear')

def calculate_obv(close, volume):
    """OBV 계산"""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index).interpolate(method='linear')

def calculate_vwap(df):
    """VWAP 계산"""
    vwap = (df['Close_TSLA'] * df['Volume_TSLA']).cumsum() / df['Volume_TSLA'].cumsum()
    return vwap

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Sortino Ratio 계산"""
    downside_returns = returns[returns < 0]
    expected_return = np.mean(returns)
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    return sortino_ratio

def calculate_omega_ratio(returns, threshold=0.0):
    """Omega Ratio 계산"""
    excess_returns = returns - threshold
    positive_returns = excess_returns[excess_returns > 0].sum()
    negative_returns = -excess_returns[excess_returns < 0].sum()
    omega_ratio = positive_returns / negative_returns if negative_returns > 0 else float('inf')
    return omega_ratio

def init_process():
    """데이터 초기화 및 지표 계산"""
    global data, volatility
    try:
        fear_greed_df = load_and_preprocess_data('fear_greed_2years.csv', date_column='date')
        tsla_df = load_and_preprocess_data('TSLA-history-2y.csv')
        tsll_df = load_and_preprocess_data('TSLL-history-2y.csv')
        vix_df = load_and_preprocess_data('VIX-history-2y.csv', date_column='Date', volume_column=None, header=3, names=['Date', 'Close'])
    except FileNotFoundError:
        sys.exit(1)  # 파일이 없으면 스크립트 종료

    data = merge_dataframes(tsla_df, tsll_df, fear_greed_df)

    data['RSI_TSLA'] = calculate_rsi(data['Close_TSLA'], 14)
    data['SMA50_TSLA'] = calculate_sma(data['Close_TSLA'], 50)
    data['SMA200_TSLA'] = calculate_sma(data['Close_TSLA'], 200)
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data['Close_TSLA'])
    data['Upper Band_TSLA'] = upper_band
    data['Middle Band_TSLA'] = middle_band
    data['Lower Band_TSLA'] = lower_band
    data['BB_width_TSLA'] = (upper_band - lower_band) / middle_band
    data['MACD_TSLA'], data['MACD_signal_TSLA'], data['MACD_histogram_TSLA'] = calculate_macd(data['Close_TSLA'])
    data['Volume Change_TSLA'] = data['Volume_TSLA'].pct_change()
    data['ATR_TSLA'] = calculate_atr(data, 14)
    data['Stochastic_K_TSLA'], data['Stochastic_D_TSLA'] = calculate_stochastic(data)
    data['OBV_TSLA'] = calculate_obv(data['Close_TSLA'], data['Volume_TSLA'])
    data['SMA5_TSLA'] = calculate_sma(data['Close_TSLA'], 5)
    data['SMA10_TSLA'] = calculate_sma(data['Close_TSLA'], 10)
    data['RSI5_TSLA'] = calculate_rsi(data['Close_TSLA'], 5)
    data['MACD_short_TSLA'], data['MACD_signal_short_TSLA'], data['MACD_histogram_short_TSLA'] = calculate_macd(data['Close_TSLA'], 5, 35, 5)
    data['VWAP_TSLA'] = calculate_vwap(data)
    weekly_rsi = calculate_rsi(data['Close_TSLA'].resample('W').last(), 14)
    data['Weekly RSI_TSLA'] = weekly_rsi.reindex(data.index, method='ffill').fillna(50)

    volatility = get_volatility_for_backtest(data)

def get_volatility_for_backtest(data):
    """VIX 평균 변동성 계산"""
    try:
        vix_df = pd.read_csv('VIX-history-2y.csv', skiprows=3, header=None, names=['Date', 'VIX'],
                             parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        error_message = "Error: 'VIX-history-2y.csv' file not found. Please run 'collect_market_data.py' to generate the required data files."
        print(error_message)
        logging.error(error_message)
        sys.exit(1)
    
    start_date = data.index.min()
    end_date = data.index.max()
    vix_filtered = vix_df.loc[start_date:end_date]
    vix_filtered['VIX'] = vix_filtered['VIX'].interpolate(method='linear')
    return vix_filtered['VIX'].mean()

def get_dynamic_param_ranges(volatility):
    """변동성에 따라 파라미터 범위 설정 (공격적으로 조정)"""
    if volatility > 30:  # 높은 변동성 환경
        return [
            (5, 30), (70, 95), (10, 30), (70, 90), (10, 30), (70, 90),  # fg_buy, fg_sell, daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell
            (0.5, 2.0), (0.2, 1.0), (-1.0, -0.2),  # volume_change_strong_buy, volume_change_weak_buy, volume_change_sell
            (2.5, 5.5), (1.5, 3.5), (1.5, 4.0),  # w_strong_buy, w_weak_buy 상향 조정, w_sell
            (10, 30), (80, 95),  # stochastic_buy, stochastic_sell
            (0.5, 3.0), (1.0, 4.0),  # obv_weight, bb_width_weight
            (10, 30), (70, 90),  # short_rsi_buy, short_rsi_sell
            (0.05, 0.3), (0.3, 0.5),  # bb_width_low, bb_width_high
            (1.5, 3.5), (1.0, 3.0)  # w_short_buy 상향 조정, w_short_sell
        ]
    return [  # 낮은 변동성 환경
        (15, 40), (60, 90), (15, 35), (65, 85), (15, 35), (65, 85),
        (0.4, 1.5), (0.1, 0.8), (-0.8, -0.1),
        (2.0, 4.5), (1.0, 3.0), (1.0, 3.0),  # w_strong_buy, w_weak_buy 상향 조정, w_sell
        (15, 35), (75, 90),
        (0.5, 2.5), (0.5, 2.5),
        (15, 35), (65, 85),
        (0.05, 0.25), (0.25, 0.45),
        (1.0, 3.0), (0.5, 2.5)  # w_short_buy 상향 조정
    ]

def get_rsi_trend(rsi_series, window=10):
    """RSI 추세 분석"""
    if len(rsi_series) < window:
        return "Stable"
    slope, _, _, _, _ = linregress(range(window), rsi_series[-window:])
    return "Increasing" if slope > 0.1 else "Decreasing" if slope < -0.1 else "Stable"

def adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row):
    """포트폴리오 조정"""
    total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
    required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])
    required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])

    if required_tsla_shares > holdings['TSLA']:
        buy_shares = required_tsla_shares - holdings['TSLA']
        cost = buy_shares * row['Close_TSLA'] * (1 + TRANSACTION_COST)
        if cash >= cost:
            cash -= cost
            holdings['TSLA'] += buy_shares
    elif required_tsla_shares < holdings['TSLA']:
        sell_shares = holdings['TSLA'] - required_tsla_shares
        proceeds = sell_shares * row['Close_TSLA'] * (1 - TRANSACTION_COST)
        cash += proceeds
        holdings['TSLA'] -= sell_shares

    if required_tsll_shares > holdings['TSLL']:
        buy_shares = required_tsll_shares - holdings['TSLL']
        cost = buy_shares * row['Close_TSLL'] * (1 + TRANSACTION_COST)
        if cash >= cost:
            cash -= cost
            holdings['TSLL'] += buy_shares
    elif required_tsll_shares < holdings['TSLL']:
        sell_shares = holdings['TSLL'] - required_tsll_shares
        proceeds = sell_shares * row['Close_TSLL'] * (1 - TRANSACTION_COST)
        cash += proceeds
        holdings['TSLL'] -= sell_shares

    return holdings, cash

def calculate_fitness(portfolio_values):
    """피트니스 스코어 계산 (Sharpe, Calmar, Total Return, Sortino, Omega Ratio 가중합)"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
    sortino_ratio = calculate_sortino_ratio(returns)
    omega_ratio = calculate_omega_ratio(returns)
    # 가중치 조정: Total Return과 Omega Ratio 강조
    return 0.15 * sharpe_ratio + 0.15 * calmar_ratio + 0.25 * total_return + 0.25 * sortino_ratio + 0.2 * omega_ratio

def evaluate(individual, data_subset=None):
    """백테스트 평가"""
    global data
    if data_subset is None:
        data_subset = data

    params = {
        "fg_buy": individual[0], "fg_sell": individual[1], "daily_rsi_buy": individual[2],
        "daily_rsi_sell": individual[3], "weekly_rsi_buy": individual[4], "weekly_rsi_sell": individual[5],
        "volume_change_strong_buy": individual[6], "volume_change_weak_buy": individual[7],
        "volume_change_sell": individual[8], "w_strong_buy": individual[9], "w_weak_buy": individual[10],
        "w_sell": individual[11], "stochastic_buy": individual[12], "stochastic_sell": individual[13],
        "obv_weight": individual[14], "bb_width_weight": individual[15],
        "short_rsi_buy": individual[16], "short_rsi_sell": individual[17],
        "bb_width_low": individual[18], "bb_width_high": individual[19],
        "w_short_buy": individual[20], "w_short_sell": individual[21]
    }

    cash = 100000
    holdings = {'TSLA': 0, 'TSLL': 0}
    current_tsll_weight = 0.0
    portfolio_values = []
    reasons_collector = []

    for i in range(1, len(data_subset)):
        row = data_subset.iloc[i]
        fear_greed = row['y']
        daily_rsi = row['RSI_TSLA']
        weekly_rsi = row['Weekly RSI_TSLA']
        daily_rsi_trend = get_rsi_trend(data_subset['RSI_TSLA'].iloc[max(0, i-10):i+1])
        close = row['Close_TSLA']
        sma5 = row['SMA5_TSLA']
        sma10 = row['SMA10_TSLA']
        sma50 = row['SMA50_TSLA']
        sma200 = row['SMA200_TSLA']
        macd = row['MACD_TSLA']
        macd_signal = row['MACD_signal_TSLA']
        macd_histogram = row['MACD_histogram_TSLA']
        volume_change = row['Volume Change_TSLA'] if not pd.isna(row['Volume Change_TSLA']) else 0
        atr = row['ATR_TSLA']
        lower_band = row['Lower Band_TSLA']
        upper_band = row['Upper Band_TSLA']
        stochastic_k = row['Stochastic_K_TSLA']
        stochastic_d = row['Stochastic_D_TSLA']
        obv = row['OBV_TSLA']
        obv_prev = data_subset['OBV_TSLA'].iloc[i-1] if i > 0 else obv
        bb_width = row['BB_width_TSLA']
        rsi5 = row['RSI5_TSLA']
        macd_short = row['MACD_short_TSLA']
        macd_signal_short = row['MACD_signal_short_TSLA']
        vwap = row['VWAP_TSLA']

        target_tsll_weight, reasons = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma5, sma10, sma50, sma200, macd, macd_signal, macd_histogram,
            volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, rsi5, macd_short, macd_signal_short, vwap,
            current_tsll_weight, params, row.name.strftime('%Y-%m-%d')
        )
        target_tsla_weight = 1 - target_tsll_weight

        holdings, cash = adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)
        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
        portfolio_values.append(total_value)
        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0
        reasons_collector.append(reasons)

    final_value = holdings['TSLA'] * data_subset['Close_TSLA'].iloc[-1] + holdings['TSLL'] * data_subset['Close_TSLL'].iloc[-1] + cash
    portfolio_values.append(final_value)

    fitness = calculate_fitness(portfolio_values)
    return fitness,

def calculate_max_drawdown(portfolio_values):
    """최대 손실 계산"""
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def clip_individual(ind, param_ranges):
    """파라미터 범위 클리핑"""
    for i in range(len(ind)):
        low, high = param_ranges[i]
        ind[i] = max(low, min(ind[i], high))
    return ind

def cross_validation_evaluation(individual, folds=15):  # 교차 검증 fold 수 증가
    """Time Series Split 교차 검증"""
    global data
    tscv = TimeSeriesSplit(n_splits=folds)
    fitness_scores = []

    for train_idx, test_idx in tscv.split(data):
        test_data = data.iloc[test_idx]
        fitness = evaluate(individual, test_data)[0]
        fitness_scores.append(fitness)

    return np.mean(fitness_scores),

def setup_toolbox(param_ranges):
    """DEAP 툴박스 설정"""
    for i, (low, high) in enumerate(param_ranges):
        toolbox.register(f"attr_{i}", random.uniform, low, high)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3,
                      toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7,
                      toolbox.attr_8, toolbox.attr_9, toolbox.attr_10, toolbox.attr_11,
                      toolbox.attr_12, toolbox.attr_13, toolbox.attr_14, toolbox.attr_15,
                      toolbox.attr_16, toolbox.attr_17, toolbox.attr_18, toolbox.attr_19,
                      toolbox.attr_20, toolbox.attr_21], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", cross_validation_evaluation)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.5)  # 변이 확률 증가
    toolbox.register("select", tools.selTournament, tournsize=7)  # 선택 압력 강화

def save_best_params(best_params):
    """최적 파라미터 저장"""
    data_to_save = {"version": "2.4", "parameters": best_params}
    with open("optimal_params.json", "w") as f:
        json.dump(data_to_save, f)

def main():
    """유전 알고리즘 실행"""
    global volatility
    try:
        init_process()
    except FileNotFoundError:
        return  # 파일이 없으면 종료

    param_ranges = get_dynamic_param_ranges(volatility)
    setup_toolbox(param_ranges)

    processes = min(cpu_count(), 4)
    pool = Pool(processes=processes, initializer=init_process)
    toolbox.register("map", pool.map)

    population = toolbox.population(n=POPULATION_SIZE)
    best_fitness = -np.inf
    patience_counter = 0

    for gen in tqdm(range(NUM_GENERATIONS), desc="Genetic Algorithm Progress"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.6)  # 변이 확률 증가
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
            print(f"Early stopping at generation {gen}: No improvement for {PATIENCE} generations")
            break

    best_ind = tools.selBest(population, 1)[0]
    best_params = {
        "fg_buy": best_ind[0], "fg_sell": best_ind[1], "daily_rsi_buy": best_ind[2],
        "daily_rsi_sell": best_ind[3], "weekly_rsi_buy": best_ind[4], "weekly_rsi_sell": best_ind[5],
        "volume_change_strong_buy": best_ind[6], "volume_change_weak_buy": best_ind[7],
        "volume_change_sell": best_ind[8], "w_strong_buy": best_ind[9], "w_weak_buy": best_ind[10],
        "w_sell": best_ind[11], "stochastic_buy": best_ind[12], "stochastic_sell": best_ind[13],
        "obv_weight": best_ind[14], "bb_width_weight": best_ind[15],
        "short_rsi_buy": best_ind[16], "short_rsi_sell": best_ind[17],
        "bb_width_low": best_ind[18], "bb_width_high": best_ind[19],
        "w_short_buy": best_ind[20], "w_short_sell": best_ind[21]
    }
    print("Optimal Parameters:", best_params)
    print("Maximum Fitness (Weighted Sharpe, Calmar, Total Return, Sortino, and Omega Ratio):", evaluate(best_ind)[0])

    save_best_params(best_params)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
