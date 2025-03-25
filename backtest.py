import pandas as pd
import numpy as np
import json
import random
from deap import base, creator, tools, algorithms
from bayes_opt import BayesianOptimization
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import product
import argparse
import logging
import sys
from functools import partial

# 초기 설정
initial_portfolio_value = 100000

# 로깅 설정 (stdout과 파일에 동시에 출력)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('optimization_log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

### 데이터 로드 및 전처리 ###
def load_and_prepare_data():
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    fear_greed_df.set_index('date', inplace=True)
    tsla_df.set_index('Date', inplace=True)
    tsll_df.set_index('Date', inplace=True)

    data = pd.concat([tsla_df, tsll_df.add_prefix('TSLL_'), fear_greed_df['y']], axis=1).dropna()

    data['Daily RSI'] = calculate_rsi(data['Close'], period=14)
    data['Weekly RSI'] = calculate_rsi(data['Close'].resample('W').last(), period=14).reindex(data.index, method='ffill')
    data['SMA50'] = calculate_sma(data['Close'], window=50)
    data['SMA200'] = calculate_sma(data['Close'], window=200)
    data['Upper Bollinger Band'], data['Lower Bollinger Band'] = calculate_bollinger_bands(data['Close'])
    data['Volume Change'] = data['Volume'].pct_change().fillna(0)
    data['MACD'], data['MACD_signal'] = calculate_macd(data['Close'])
    data['ATR'] = calculate_atr(data, period=14)

    return data

### 지표 계산 함수 ###
def calculate_rsi(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = data.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = data.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, macd_signal

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

### 벡터화된 백테스팅 함수 ###
def simulate_backtest_vectorized(data, params):
    portfolio_value = np.ones(len(data)) * initial_portfolio_value
    current_tsll_weight = np.zeros(len(data))

    # 동적 임계값 계산 (ATR 기반)
    atr_normalized = data['ATR'] / data['Close']
    volume_change_strong_buy = params['volume_change_strong_buy'] * (1 + atr_normalized)
    volume_change_weak_buy = params['volume_change_weak_buy'] * (1 + atr_normalized)
    volume_change_sell = params['volume_change_sell'] * (1 + atr_normalized)

    # Buy 및 Sell 조건 (세분화 포함)
    strong_buy_conditions = (
        (data['Daily RSI'] < params['daily_rsi_buy']) &
        (data['Weekly RSI'] < params['weekly_rsi_buy']) &
        (data['Close'] > data['Lower Bollinger Band']) &
        (data['Volume Change'] > volume_change_strong_buy) &
        (data['y'] < params['fg_buy']) &
        (data['MACD'] > data['MACD_signal'])
    )
    weak_buy_conditions = (
        (data['Daily RSI'] < params['daily_rsi_buy']) &
        (data['Weekly RSI'] < params['weekly_rsi_buy']) &
        (data['Volume Change'] > volume_change_weak_buy) &
        (data['Volume Change'] <= volume_change_strong_buy) &  # Strong Buy와 구분
        (data['y'] < params['fg_buy']) &
        (data['MACD'] > data['MACD_signal'])
    )
    sell_conditions = (
        (data['Daily RSI'] > params['daily_rsi_sell']) &
        (data['Weekly RSI'] > params['weekly_rsi_sell']) &
        (data['Close'] < data['Upper Bollinger Band']) &
        (data['Volume Change'] < volume_change_sell) &
        (data['y'] > params['fg_sell']) &
        (data['MACD'] < data['MACD_signal'])
    )

    # 가중치 적용
    w_strong_buy = params['w_strong_buy']
    w_weak_buy = params['w_weak_buy']
    w_sell = params['w_sell']

    strong_buy_adj = w_strong_buy * strong_buy_conditions * 0.1
    weak_buy_adj = w_weak_buy * weak_buy_conditions * 0.1
    sell_adj = w_sell * sell_conditions * 0.1

    target_weight = np.clip(current_tsll_weight + strong_buy_adj + weak_buy_adj - sell_adj, 0, 0.8)
    current_tsll_weight = target_weight

    tsla_return = data['Close'].pct_change().fillna(0)
    tsll_return = data['TSLL_Close'].pct_change().fillna(0)
    portfolio_return = current_tsll_weight.shift(1) * tsll_return + (1 - current_tsll_weight.shift(1)) * tsla_return
    portfolio_value = initial_portfolio_value * (1 + portfolio_return).cumprod()

    return portfolio_value.iloc[-1]

### Grid Search를 위한 전역 평가 함수 ###
def evaluate_combination(comb, data, keys):
    params = dict(zip(keys, comb))
    penalty = 0
    if params['daily_rsi_sell'] - params['daily_rsi_buy'] < 10:
        penalty += (10 - (params['daily_rsi_sell'] - params['daily_rsi_buy'])) * 10000
    if params['weekly_rsi_sell'] - params['weekly_rsi_buy'] < 10:
        penalty += (10 - (params['weekly_rsi_sell'] - params['weekly_rsi_buy'])) * 10000
    if params['fg_sell'] - params['fg_buy'] < 10:
        penalty += (10 - (params['fg_sell'] - params['fg_buy'])) * 10000
    fitness = simulate_backtest_vectorized(data, params) - penalty
    return fitness

### Grid Search 최적화 함수 ###
def optimize_grid_search_parallel(data, processes, n_samples):
    param_grid = {
        'daily_rsi_buy': [10, 20, 30, 40],
        'daily_rsi_sell': [60, 70, 80, 90],
        'weekly_rsi_buy': [20, 30, 40, 50],
        'weekly_rsi_sell': [60, 70, 80, 90],
        'fg_buy': [10, 20, 30, 40, 50],
        'fg_sell': [60, 70, 80, 90],
        'volume_change_strong_buy': [0.4, 0.5, 0.6, 0.7],
        'volume_change_weak_buy': [0.1, 0.2, 0.3, 0.4],
        'volume_change_sell': [-0.4, -0.3, -0.2, -0.1],
        'w_strong_buy': [1.5, 2.0, 2.5, 3.0],
        'w_weak_buy': [0.5, 1.0, 1.5, 2.0],
        'w_sell': [0.5, 1.0, 1.5, 2.0]
    }

    keys = list(param_grid.keys())
    all_combinations = list(product(*[param_grid[key] for key in keys]))
    if len(all_combinations) < n_samples:
        logger.warning(f"Total combinations ({len(all_combinations)}) are less than requested samples ({n_samples}). Using all combinations.")
        sampled_combinations = all_combinations
    else:
        sampled_combinations = random.sample(all_combinations, n_samples)

    logger.info(f"Grid Search 시작: {len(sampled_combinations)}개의 샘플링된 조합, CPU 코어 수: {processes}")

    best_value = -np.inf
    best_params = None

    start_time = time.time()
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.starmap(evaluate_combination, [(comb, data, keys) for comb in sampled_combinations]), total=len(sampled_combinations), desc="Grid Search Progress"))
        for i, (comb, value) in enumerate(zip(sampled_combinations, results)):
            if value > best_value:
                best_value = value
                best_params = dict(zip(keys, comb))
            if i % 1000 == 0 and i > 0:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / (i + 1)) * len(sampled_combinations)
                remaining_time = estimated_total_time - elapsed_time
                logger.info(f"Grid Search 진행 중: {i}/{len(sampled_combinations)}, 예상 종료까지 {remaining_time:.2f}초")

    logger.info(f"Grid Search 완료: 최적 파라미터 {best_params}, 최종 포트폴리오 가치 {best_value:.2f}")
    return best_params, best_value

### GA 최적화 함수 ###
def evaluate(individual, data):
    params = {
        'daily_rsi_buy': individual[0],
        'daily_rsi_sell': individual[1],
        'weekly_rsi_buy': individual[2],
        'weekly_rsi_sell': individual[3],
        'fg_buy': individual[4],
        'fg_sell': individual[5],
        'volume_change_strong_buy': individual[6],
        'volume_change_weak_buy': individual[7],
        'volume_change_sell': individual[8],
        'w_strong_buy': individual[9],
        'w_weak_buy': individual[10],
        'w_sell': individual[11]
    }
    penalty = 0
    if params['daily_rsi_sell'] - params['daily_rsi_buy'] < 10:
        penalty += (10 - (params['daily_rsi_sell'] - params['daily_rsi_buy'])) * 10000
    if params['weekly_rsi_sell'] - params['weekly_rsi_buy'] < 10:
        penalty += (10 - (params['weekly_rsi_sell'] - params['weekly_rsi_buy'])) * 10000
    if params['fg_sell'] - params['fg_buy'] < 10:
        penalty += (10 - (params['fg_sell'] - params['fg_buy'])) * 10000
    if params['volume_change_strong_buy'] <= params['volume_change_weak_buy']:
        penalty += 10000  # Strong Buy가 Weak Buy보다 커야 함
    fitness = simulate_backtest_vectorized(data, params) - penalty
    return (fitness,)

def clip_individual(individual, param_ranges):
    for i, (low, high) in enumerate(param_ranges):
        individual[i] = max(low, min(high, individual[i]))
    return individual

def generate_individual(toolbox, param_ranges):
    while True:
        individual = toolbox.individual()
        params = {
            'daily_rsi_buy': individual[0],
            'daily_rsi_sell': individual[1],
            'weekly_rsi_buy': individual[2],
            'weekly_rsi_sell': individual[3],
            'fg_buy': individual[4],
            'fg_sell': individual[5],
            'volume_change_strong_buy': individual[6],
            'volume_change_weak_buy': individual[7],
            'volume_change_sell': individual[8],
            'w_strong_buy': individual[9],
            'w_weak_buy': individual[10],
            'w_sell': individual[11]
        }
        if (params['daily_rsi_sell'] - params['daily_rsi_buy'] >= 10 and
            params['weekly_rsi_sell'] - params['weekly_rsi_buy'] >= 10 and
            params['fg_sell'] - params['fg_buy'] >= 10 and
            params['volume_change_strong_buy'] > params['volume_change_weak_buy']):
            return individual

def optimize_ga_parallel(data, processes, population_size, generations):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    param_ranges = [
        (0, 50),     # daily_rsi_buy
        (50, 100),   # daily_rsi_sell
        (0, 50),     # weekly_rsi_buy
        (50, 100),   # weekly_rsi_sell
        (0, 50),     # fg_buy
        (50, 100),   # fg_sell
        (0.3, 1.0),  # volume_change_strong_buy
        (0.1, 0.5),  # volume_change_weak_buy
        (-0.5, -0.1),# volume_change_sell
        (1.0, 3.0),  # w_strong_buy
        (0.5, 2.0),  # w_weak_buy
        (0.5, 2.0)   # w_sell
    ]
    for i, (low, high) in enumerate(param_ranges):
        toolbox.register(f"attr{i}", random.uniform, low, high)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.attr0, toolbox.attr1, toolbox.attr2, toolbox.attr3,
                      toolbox.attr4, toolbox.attr5, toolbox.attr6, toolbox.attr7,
                      toolbox.attr8, toolbox.attr9, toolbox.attr10, toolbox.attr11], n=1)
    toolbox.register("population", tools.initRepeat, list, partial(generate_individual, toolbox, param_ranges))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate_with_clip(individual):
        tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)
        clip_individual(individual, param_ranges)
        return individual,

    toolbox.register("mutate", mutate_with_clip)
    toolbox.register("evaluate", partial(evaluate, data=data))

    pool = Pool(processes=processes)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logger.info(f"GA 시작: CPU 코어 수: {processes}, 개체군 크기: {population_size}, 세대 수: {generations}")
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations,
                                       stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_params = {
        'daily_rsi_buy': max(0, min(50, best_individual[0])),
        'daily_rsi_sell': max(50, min(100, best_individual[1])),
        'weekly_rsi_buy': max(0, min(50, best_individual[2])),
        'weekly_rsi_sell': max(50, min(100, best_individual[3])),
        'fg_buy': max(0, min(50, best_individual[4])),
        'fg_sell': max(50, min(100, best_individual[5])),
        'volume_change_strong_buy': max(0.3, min(1.0, best_individual[6])),
        'volume_change_weak_buy': max(0.1, min(0.5, best_individual[7])),
        'volume_change_sell': max(-0.5, min(-0.1, best_individual[8])),
        'w_strong_buy': max(1.0, min(3.0, best_individual[9])),
        'w_weak_buy': max(0.5, min(2.0, best_individual[10])),
        'w_sell': max(0.5, min(2.0, best_individual[11]))
    }
    best_value = evaluate(best_individual, data)[0]
    logger.info(f"GA 완료: 최적 파라미터 {best_params}, 최종 포트폴리오 가치 {best_value:.2f}")
    return best_params, best_value, logbook

### 베이지안 최적화 함수 ###
def optimize_bayesian(data, init_points, n_iter):
    def black_box_function(daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell, fg_buy, fg_sell,
                           volume_change_strong_buy, volume_change_weak_buy, volume_change_sell,
                           w_strong_buy, w_weak_buy, w_sell):
        params = {
            'daily_rsi_buy': daily_rsi_buy,
            'daily_rsi_sell': daily_rsi_sell,
            'weekly_rsi_buy': weekly_rsi_buy,
            'weekly_rsi_sell': weekly_rsi_sell,
            'fg_buy': fg_buy,
            'fg_sell': fg_sell,
            'volume_change_strong_buy': volume_change_strong_buy,
            'volume_change_weak_buy': volume_change_weak_buy,
            'volume_change_sell': volume_change_sell,
            'w_strong_buy': w_strong_buy,
            'w_weak_buy': w_weak_buy,
            'w_sell': w_sell
        }
        penalty = 0
        if daily_rsi_sell - daily_rsi_buy < 10:
            penalty += (10 - (daily_rsi_sell - daily_rsi_buy)) * 10000
        if weekly_rsi_sell - weekly_rsi_buy < 10:
            penalty += (10 - (weekly_rsi_sell - weekly_rsi_buy)) * 10000
        if fg_sell - fg_buy < 10:
            penalty += (10 - (fg_sell - fg_buy)) * 10000
        if volume_change_strong_buy <= volume_change_weak_buy:
            penalty += 10000
        fitness = simulate_backtest_vectorized(data, params) - penalty
        return fitness

    pbounds = {
        'daily_rsi_buy': (0, 50),
        'daily_rsi_sell': (50, 100),
        'weekly_rsi_buy': (0, 50),
        'weekly_rsi_sell': (50, 100),
        'fg_buy': (0, 50),
        'fg_sell': (50, 100),
        'volume_change_strong_buy': (0.3, 1.0),
        'volume_change_weak_buy': (0.1, 0.5),
        'volume_change_sell': (-0.5, -0.1),
        'w_strong_buy': (1.0, 3.0),
        'w_weak_buy': (0.5, 2.0),
        'w_sell': (0.5, 2.0)
    }

    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=1)
    logger.info(f"베이지안 최적화 시작: 초기 포인트 {init_points}개, 반복 {n_iter}회")
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best_params = optimizer.max['params']
    best_value = optimizer.max['target']
    logger.info(f"베이지안 최적화 완료: 최적 파라미터 {best_params}, 최종 포트폴리오 가치 {best_value:.2f}")
    return best_params, best_value

### 최적 파라미터 저장 함수 ###
def save_optimal_params(best_params, file_path="optimal_params.json"):
    with open(file_path, "w") as f:
        json.dump(best_params, f)
    logger.info(f"최적 파라미터가 {file_path}에 저장되었습니다.")

### 메인 함수 ###
def main(args):
    data = load_and_prepare_data()
    method = args.method
    processes = args.processes if args.processes else cpu_count()

    if method == 'grid_search':
        n_samples = args.n_samples if args.n_samples else 100000
        best_params, best_value = optimize_grid_search_parallel(data, processes=processes, n_samples=n_samples)
        logbook = None
    elif method == 'ga':
        population_size = args.population_size if args.population_size else 100
        generations = args.generations if args.generations else 50
        best_params, best_value, logbook = optimize_ga_parallel(data, processes=processes, population_size=population_size, generations=generations)
    elif method == 'bayesian':
        init_points = args.init_points if args.init_points else 10
        n_iter = args.n_iter if args.n_iter else 50
        best_params, best_value = optimize_bayesian(data, init_points=init_points, n_iter=n_iter)
        logbook = None
    else:
        logger.error("Invalid optimization method selected.")
        return

    elapsed_time = time.time() - start_time
    logger.info(f"{method.upper()} 완료: 실행 시간 {elapsed_time:.2f}초, 최종 포트폴리오 가치 {best_value:.2f}")

    if method == 'ga' and logbook:
        logger.info("\nGA 세대별 통계 정보:")
        for record in logbook:
            logger.info(f"세대 {record['gen']}: 평균 {record['avg']:.2f}, 최소 {record['min']:.2f}, 최대 {record['max']:.2f}")

    save_optimal_params(best_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Optimization")
    parser.add_argument('--method', type=str, choices=['grid_search', 'ga', 'bayesian'], required=True, help="Optimization method to use")
    parser.add_argument('--processes', type=int, help="Number of processes to use for parallelization")
    parser.add_argument('--n_samples', type=int, help="Number of samples for Grid Search (default: 100,000)")
    parser.add_argument('--population_size', type=int, help="Population size for GA (default: 100)")
    parser.add_argument('--generations', type=int, help="Number of generations for GA (default: 50)")
    parser.add_argument('--init_points', type=int, help="Initial points for Bayesian Optimization (default: 10)")
    parser.add_argument('--n_iter', type=int, help="Number of iterations for Bayesian Optimization (default: 50)")

    args = parser.parse_args()
    start_time = time.time()
    main(args)
