import pandas as pd
import numpy as np
from itertools import product
import json
import random
from deap import base, creator, tools, algorithms
from bayes_opt import BayesianOptimization

# 초기 설정
initial_portfolio_value = 100000

### 데이터 로드 및 전처리 ###
def load_and_prepare_data():
    """
    과거 데이터를 로드하고 지표를 계산하는 함수.
    실제 구현 시 사용자 데이터에 맞게 수정 필요.
    """
    # 데이터 로드 (예시 경로, 실제 경로로 대체 필요)
    fear_greed_df = pd.read_csv('fear_greed_2years.csv', parse_dates=['date'])
    tsla_df = pd.read_csv('TSLA-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')
    tsll_df = pd.read_csv('TSLL-history-2y.csv', parse_dates=['Date'], date_format='%m/%d/%Y')

    # 데이터 전처리
    tsla_df['Volume'] = tsla_df['Volume'].str.replace(',', '').astype(float)
    tsll_df['Volume'] = tsll_df['Volume'].str.replace(',', '').astype(float)

    # 날짜 형식 통일 및 인덱스 설정
    fear_greed_df.set_index('date', inplace=True)
    tsla_df.set_index('Date', inplace=True)
    tsll_df.set_index('Date', inplace=True)

    # 데이터 병합
    data = pd.concat([tsla_df, tsll_df.add_prefix('TSLL_'), fear_greed_df['y']], axis=1).dropna()

    # 지표 계산
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

### 백테스팅 함수 ###
def simulate_backtest(data, params):
    """
    주어진 파라미터로 백테스팅을 수행하여 최종 포트폴리오 가치를 계산.
    """
    portfolio_value = initial_portfolio_value
    current_tsll_weight = 0.0

    for t in range(len(data) - 1):
        row = data.iloc[t]
        buy_conditions = [
            row['Daily RSI'] < params['daily_rsi_buy'],
            row['Weekly RSI'] < params['weekly_rsi_buy'],
            row['Close'] > row['Lower Bollinger Band'],
            row['Volume Change'] > params['volume_change_buy'],
            row['y'] < params['fg_buy'],
            row['MACD'] > row['MACD_signal']
        ]
        sell_conditions = [
            row['Daily RSI'] > params['daily_rsi_sell'],
            row['Weekly RSI'] > params['weekly_rsi_sell'],
            row['Close'] < row['Upper Bollinger Band'],
            row['Volume Change'] < params['volume_change_sell'],
            row['y'] > params['fg_sell'],
            row['MACD'] < row['MACD_signal']
        ]

        # 변동성에 따른 가중치 동적 조정
        atr = row['ATR']
        w_buy = params['w_buy'] / (1 + atr / data['Close'].mean())
        w_sell = params['w_sell'] * (1 + atr / data['Close'].mean())

        # 비중 조정
        base_weight = current_tsll_weight
        buy_adjustment = w_buy * sum(buy_conditions) * 0.1
        sell_adjustment = w_sell * sum(sell_conditions) * 0.1
        target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 0.8))
        current_tsll_weight = target_weight

        # 수익률 계산
        tsla_return = (data['Close'].iloc[t + 1] / data['Close'].iloc[t]) - 1
        tsll_return = (data['TSLL_Close'].iloc[t + 1] / data['TSLL_Close'].iloc[t]) - 1
        portfolio_return = current_tsll_weight * tsll_return + (1 - current_tsll_weight) * tsla_return
        portfolio_value *= (1 + portfolio_return)

    return portfolio_value

### 그리드 서치 최적화 함수 ###
def optimize_grid_search(data):
    """
    그리드 서치를 통해 최적의 파라미터를 탐색.
    """
    best_value = 0
    best_params = {}

    param_grid = {
        'daily_rsi_buy': [20, 25, 30, 35, 40],
        'daily_rsi_sell': [60, 65, 70, 75, 80],
        'weekly_rsi_buy': [30, 35, 40, 45, 50],
        'weekly_rsi_sell': [50, 55, 60, 65, 70],
        'fg_buy': [20, 30, 40, 50, 60],
        'fg_sell': [60, 65, 70, 75, 80],
        'volume_change_buy': [0.05, 0.1, 0.15, 0.2],
        'volume_change_sell': [-0.2, -0.15, -0.1, -0.05],
        'w_buy': [0.5, 1.0, 1.5, 2.0, 2.5],
        'w_sell': [0.5, 1.0, 1.5, 2.0, 2.5]
    }

    keys = list(param_grid.keys())
    combinations = product(*[param_grid[key] for key in keys])

    for combination in combinations:
        params = dict(zip(keys, combination))
        final_value = simulate_backtest(data, params)
        if final_value > best_value:
            best_value = final_value
            best_params = params

    return best_params, best_value

### 유전 알고리즘 최적화 함수 ###
def optimize_ga(data):
    """
    유전 알고리즘을 통해 최적의 파라미터를 탐색.
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    param_ranges = {
        'daily_rsi_buy': (20, 40),
        'daily_rsi_sell': (60, 80),
        'weekly_rsi_buy': (30, 50),
        'weekly_rsi_sell': (50, 70),
        'fg_buy': (20, 60),
        'fg_sell': (60, 80),
        'volume_change_buy': (0.05, 0.2),
        'volume_change_sell': (-0.2, -0.05),
        'w_buy': (0.5, 2.5),
        'w_sell': (0.5, 2.5)
    }

    def evaluate(individual):
        params = dict(zip(param_ranges.keys(), individual))
        return simulate_backtest(data, params),

    toolbox = base.Toolbox()
    for param, (low, high) in param_ranges.items():
        toolbox.register(param, random.uniform, low, high)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [getattr(toolbox, param) for param in param_ranges.keys()], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                 halloffame=hof, verbose=False)

    best_params = dict(zip(param_ranges.keys(), hof[0]))
    best_value = simulate_backtest(data, best_params)
    return best_params, best_value

### 베이지안 최적화 함수 ###
def optimize_bayesian(data):
    """
    베이지안 최적화를 통해 최적의 파라미터를 탐색.
    """
    def black_box_function(**params):
        return simulate_backtest(data, params)

    pbounds = {
        'daily_rsi_buy': (20, 40),
        'daily_rsi_sell': (60, 80),
        'weekly_rsi_buy': (30, 50),
        'weekly_rsi_sell': (50, 70),
        'fg_buy': (20, 60),
        'fg_sell': (60, 80),
        'volume_change_buy': (0.05, 0.2),
        'volume_change_sell': (-0.2, -0.05),
        'w_buy': (0.5, 2.5),
        'w_sell': (0.5, 2.5)
    }

    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points=5, n_iter=25)

    best_params = optimizer.max['params']
    best_value = optimizer.max['target']
    return best_params, best_value

### 최적 파라미터 저장 함수 ###
def save_optimal_params(best_params, file_path="optimal_params.json"):
    """
    최적 파라미터를 JSON 파일로 저장.
    """
    with open(file_path, "w") as f:
        json.dump(best_params, f)
    print(f"최적 파라미터가 {file_path}에 저장되었습니다.")

### 메인 함수 ###
def main():
    data = load_and_prepare_data()

    # 최적화 방법 리스트
    methods = ['grid_search', 'ga', 'bayesian']
    results = {}

    # 각 방법으로 최적화 수행
    for method in methods:
        if method == 'grid_search':
            best_params, best_value = optimize_grid_search(data)
        elif method == 'ga':
            best_params, best_value = optimize_ga(data)
        elif method == 'bayesian':
            best_params, best_value = optimize_bayesian(data)

        results[method] = {
            'params': best_params,
            'value': best_value,
            'return': (best_value - initial_portfolio_value) / initial_portfolio_value * 100
        }

    # 최적 방법 선택
    best_method = max(results, key=lambda x: results[x]['value'])
    best_params = results[best_method]['params']
    best_value = results[best_method]['value']
    best_return = results[best_method]['return']

    # 결과 출력
    print(f"최적의 방법: {best_method}")
    print(f"최적 파라미터: {best_params}")
    print(f"최종 포트폴리오 가치: ${best_value:.2f}")
    print(f"수익률: {best_return:.2f}%")

    # 최적 파라미터 저장
    save_optimal_params(best_params)

if __name__ == "__main__":
    main()
