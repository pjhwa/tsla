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

# 초기 설정
initial_portfolio_value = 100000

# 전역 변수로 데이터와 keys 설정
global_data = None
global_keys = None

# 로깅 설정 (stdout과 파일에 동시에 출력)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('optimization_log', mode='a')  # 'a' 모드로 추가 기록
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

### 데이터 로드 및 전처리 ###
def load_and_prepare_data():
    """
    과거 데이터를 로드하고 지표를 계산하는 함수.
    실제 구현 시 사용자 데이터에 맞게 수정 필요.
    """
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
    """RSI (Relative Strength Index)를 계산"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    """단순 이동 평균(SMA)을 계산"""
    return data.rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    """볼린저 밴드(Bollinger Bands)를 계산"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD (Moving Average Convergence Divergence)를 계산"""
    ema_fast = data.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = data.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, macd_signal

def calculate_atr(data, period=14):
    """평균 진폭 범위(ATR, Average True Range)를 계산"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

### 벡터화된 백테스팅 함수 ###
def simulate_backtest_vectorized(data, params):
    """
    주어진 파라미터로 백테스팅을 수행하여 최종 포트폴리오 가치를 계산 (벡터화).
    """
    portfolio_value = np.ones(len(data)) * initial_portfolio_value
    current_tsll_weight = np.zeros(len(data))

    # 매수/매도 조건을 벡터화
    buy_conditions = (
        (data['Daily RSI'] < params['daily_rsi_buy']) &
        (data['Weekly RSI'] < params['weekly_rsi_buy']) &
        (data['Close'] > data['Lower Bollinger Band']) &
        (data['Volume Change'] > params['volume_change_buy']) &
        (data['y'] < params['fg_buy']) &
        (data['MACD'] > data['MACD_signal'])
    )
    sell_conditions = (
        (data['Daily RSI'] > params['daily_rsi_sell']) &
        (data['Weekly RSI'] > params['weekly_rsi_sell']) &
        (data['Close'] < data['Upper Bollinger Band']) &
        (data['Volume Change'] < params['volume_change_sell']) &
        (data['y'] > params['fg_sell']) &
        (data['MACD'] < data['MACD_signal'])
    )

    # ATR을 사용한 가중치 계산
    atr = data['ATR']
    w_buy = params['w_buy'] / (1 + atr / data['Close'].mean())
    w_sell = params['w_sell'] * (1 + atr / data['Close'].mean())

    # 비중 조정
    buy_adj = w_buy * buy_conditions * 0.1
    sell_adj = w_sell * sell_conditions * 0.1
    target_weight = np.clip(current_tsll_weight + buy_adj - sell_adj, 0, 0.8)
    current_tsll_weight = target_weight

    # 수익률 계산
    tsla_return = data['Close'].pct_change().fillna(0)
    tsll_return = data['TSLL_Close'].pct_change().fillna(0)
    portfolio_return = current_tsll_weight.shift(1) * tsll_return + (1 - current_tsll_weight.shift(1)) * tsla_return
    portfolio_value = initial_portfolio_value * (1 + portfolio_return).cumprod()

    return portfolio_value.iloc[-1]

### Grid Search를 위한 전역 평가 함수 ###
def evaluate_combination(comb):
    """
    Grid Search에서 사용할 전역 평가 함수.
    global_data와 global_keys를 사용하여 데이터와 파라미터 키에 접근.
    """
    global global_data, global_keys
    params = dict(zip(global_keys, comb))
    return simulate_backtest_vectorized(global_data, params)

### Grid Search 최적화 함수 (병렬 처리 및 진행 상황 표시) ###
def optimize_grid_search_parallel(data, processes, n_samples):
    """
    병렬 Grid Search를 통해 최적의 파라미터를 탐색 (랜덤 샘플링).
    """
    global global_data, global_keys
    global_data = data  # 전역 변수에 데이터 할당

    param_grid = {
        'daily_rsi_buy': [10, 20, 30, 40],
        'daily_rsi_sell': [60, 70, 80, 90],
        'weekly_rsi_buy': [20, 30, 40, 50],
        'weekly_rsi_sell': [50, 60, 70, 80],
        'fg_buy': [10, 20, 30, 40, 50, 60],
        'fg_sell': [60, 70, 80, 90],
        'volume_change_buy': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'volume_change_sell': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'w_buy': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'w_sell': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    }

    global_keys = list(param_grid.keys())  # 전역 변수에 keys 설정
    all_combinations = list(product(*[param_grid[key] for key in global_keys]))
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
        results = list(tqdm(pool.imap(evaluate_combination, sampled_combinations), total=len(sampled_combinations), desc="Grid Search Progress"))
        for i, (comb, value) in enumerate(zip(sampled_combinations, results)):
            if value > best_value:
                best_value = value
                best_params = dict(zip(global_keys, comb))
            if i % 1000 == 0 and i > 0:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / (i + 1)) * len(sampled_combinations)
                remaining_time = estimated_total_time - elapsed_time
                logger.info(f"Grid Search 진행 중: {i}/{len(sampled_combinations)}, 예상 종료까지 {remaining_time:.2f}초")

    logger.info(f"Grid Search 완료: 최적 파라미터 {best_params}, 최종 포트폴리오 가치 {best_value:.2f}")
    return best_params, best_value

### GA 최적화 함수 (병렬 처리) ###
def evaluate(individual):
    """
    개체의 적합도를 평가하는 전역 함수.
    """
    global global_data
    params = {
        'daily_rsi_buy': individual[0],
        'daily_rsi_sell': individual[1],
        'weekly_rsi_buy': individual[2],
        'weekly_rsi_sell': individual[3],
        'fg_buy': individual[4],
        'fg_sell': individual[5],
        'volume_change_buy': individual[6],
        'volume_change_sell': individual[7],
        'w_buy': individual[8],
        'w_sell': individual[9]
    }
    return simulate_backtest_vectorized(global_data, params),

def optimize_ga_parallel(data, processes, population_size, generations):
    """
    병렬 유전 알고리즘을 통해 최적의 파라미터를 탐색.
    """
    global global_data
    global_data = data  # 전역 변수에 데이터 할당

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    param_ranges = [
        (10, 40), (60, 90), (20, 50), (50, 80),  # RSI
        (10, 60), (60, 90),  # Fear & Greed
        (-1.0, 1.0), (-1.0, 1.0),  # Volume Change
        (0.5, 3.0), (0.5, 3.0)  # Weights
    ]
    for i, (low, high) in enumerate(param_ranges):
        toolbox.register(f"attr{i}", random.uniform, low, high)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.attr0, toolbox.attr1, toolbox.attr2, toolbox.attr3,
                      toolbox.attr4, toolbox.attr5, toolbox.attr6, toolbox.attr7,
                      toolbox.attr8, toolbox.attr9], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

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

    best_params = {
        'daily_rsi_buy': hof[0][0], 'daily_rsi_sell': hof[0][1],
        'weekly_rsi_buy': hof[0][2], 'weekly_rsi_sell': hof[0][3],
        'fg_buy': hof[0][4], 'fg_sell': hof[0][5],
        'volume_change_buy': max(min(hof[0][6], 1.0), -1.0),
        'volume_change_sell': max(min(hof[0][7], 1.0), -1.0),
        'w_buy': hof[0][8], 'w_sell': hof[0][9]
    }
    best_value = evaluate(hof[0])[0]
    logger.info(f"GA 완료: 최적 파라미터 {best_params}, 최종 포트폴리오 가치 {best_value:.2f}")
    return best_params, best_value, logbook

### 베이지안 최적화 함수 ###
def optimize_bayesian(data, init_points, n_iter):
    """
    베이지안 최적화를 통해 최적의 파라미터를 탐색.
    """
    def black_box_function(daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell, fg_buy, fg_sell, volume_change_buy, volume_change_sell, w_buy, w_sell):
        params = {
            'daily_rsi_buy': daily_rsi_buy,
            'daily_rsi_sell': daily_rsi_sell,
            'weekly_rsi_buy': weekly_rsi_buy,
            'weekly_rsi_sell': weekly_rsi_sell,
            'fg_buy': fg_buy,
            'fg_sell': fg_sell,
            'volume_change_buy': volume_change_buy,
            'volume_change_sell': volume_change_sell,
            'w_buy': w_buy,
            'w_sell': w_sell
        }
        return simulate_backtest_vectorized(data, params)

    pbounds = {
        'daily_rsi_buy': (10, 40),
        'daily_rsi_sell': (60, 90),
        'weekly_rsi_buy': (20, 50),
        'weekly_rsi_sell': (50, 80),
        'fg_buy': (10, 60),
        'fg_sell': (60, 90),
        'volume_change_buy': (-1.0, 1.0),
        'volume_change_sell': (-1.0, 1.0),
        'w_buy': (0.5, 3.0),
        'w_sell': (0.5, 3.0)
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
    """
    최적 파라미터를 JSON 파일로 저장.
    """
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

    # 세대별 통계 정보 출력 (GA의 경우)
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
