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
POPULATION_SIZE = 1000  # 탐색 범위 확대
NUM_GENERATIONS = 300  # 최적화 심화
PATIENCE = 50  # 조기 종료 인내심 증가
TRANSACTION_COST = 0.003  # 거래 비용 현실화 (0.3%)
MAX_POSITION_SIZE = 0.8  # 최대 포지션 크기 제한 (80%)
STOP_LOSS_THRESHOLD = -0.05  # 손절매 기준 (-5%)

# 전역 변수
data = None
volatility = None

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path, date_column='Date', volume_column='Volume', header=0, names=None):
    """CSV 파일 로드 및 전처리"""
    if not os.path.exists(file_path):
        error_message = f"Error: '{file_path}' file not found."
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
    """스토캐스틱 계산"""
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
    return (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

def calculate_calmar_ratio(total_return, max_drawdown):
    """Calmar Ratio 계산"""
    return total_return / abs(max_drawdown) if max_drawdown < 0 else 1e6

def calculate_max_drawdown(portfolio_values):
    """최대 손실 계산"""
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / peak
    return -np.max(drawdowns) if len(drawdowns) > 0 else 0

def calculate_fitness(portfolio_values):
    """피트니스 함수 개선"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    calmar_ratio = calculate_calmar_ratio(total_return, max_drawdown)
    sortino_ratio = calculate_sortino_ratio(returns)
    # 하방 리스크와 수익률 강조
    fitness = (0.1 * sharpe_ratio + 0.3 * calmar_ratio + 0.3 * total_return + 0.3 * sortino_ratio)
    return fitness

def init_process():
    """데이터 초기화 및 지표 계산"""
    global data, volatility
    try:
        fear_greed_df = load_and_preprocess_data('fear_greed_2years.csv', date_column='date')
        tsla_df = load_and_preprocess_data('TSLA-history-2y.csv')
        tsll_df = load_and_preprocess_data('TSLL-history-2y.csv')
        vix_df = load_and_preprocess_data('VIX-history-2y.csv', date_column='Date', volume_column=None, header=3, names=['Date', 'Close'])
    except FileNotFoundError:
        sys.exit(1)

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
    data['MACD_short_TSLA'], data['MACD_signal_short_TSLA'], _ = calculate_macd(data['Close_TSLA'], 5, 35, 5)
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
        logging.error("VIX file not found.")
        sys.exit(1)
    
    start_date = data.index.min()
    end_date = data.index.max()
    vix_filtered = vix_df.loc[start_date:end_date]
    vix_filtered['VIX'] = vix_filtered['VIX'].interpolate(method='linear')
    return vix_filtered['VIX'].mean()

def get_dynamic_param_ranges(volatility):
    """변동성 기반 동적 파라미터 범위"""
    if volatility > 30:  # 고변동성 환경
        return [
            (10, 35), (65, 90), (15, 35), (65, 85), (20, 40), (60, 80),  # fg_buy, fg_sell, daily_rsi_buy, daily_rsi_sell, weekly_rsi_buy, weekly_rsi_sell
            (0.6, 2.0), (0.2, 1.0), (-1.0, -0.3),  # volume_change_strong_buy, volume_change_weak_buy, volume_change_sell
            (2.0, 4.0), (1.0, 2.5), (1.5, 3.5),  # w_strong_buy, w_weak_buy, w_sell
            (15, 30), (75, 90),  # stochastic_buy, stochastic_sell
            (0.5, 2.0), (0.5, 2.0),  # obv_weight, bb_width_weight
            (20, 35), (65, 80),  # short_rsi_buy, short_rsi_sell
            (0.1, 0.3), (0.25, 0.5),  # bb_width_low, bb_width_high
            (1.0, 2.5), (1.0, 2.5)  # w_short_buy, w_short_sell
        ]
    else:  # 저변동성 환경
        return [
            (15, 40), (60, 85), (20, 40), (60, 80), (25, 45), (55, 75),
            (0.4, 1.5), (0.1, 0.8), (-0.8, -0.1),
            (1.5, 3.5), (0.8, 2.0), (1.0, 3.0),
            (20, 35), (70, 85),
            (0.3, 1.5), (0.3, 1.5),
            (25, 40), (60, 75),
            (0.05, 0.2), (0.15, 0.4),
            (0.8, 2.0), (0.8, 2.0)
        ]

def evaluate_individual(individual, fold_data):
    """개별 파라미터 평가"""
    params = {
        "fg_buy": individual[0], "fg_sell": individual[1], "daily_rsi_buy": individual[2], "daily_rsi_sell": individual[3],
        "weekly_rsi_buy": individual[4], "weekly_rsi_sell": individual[5], "volume_change_strong_buy": individual[6],
        "volume_change_weak_buy": individual[7], "volume_change_sell": individual[8], "w_strong_buy": individual[9],
        "w_weak_buy": individual[10], "w_sell": individual[11], "stochastic_buy": individual[12], "stochastic_sell": individual[13],
        "obv_weight": individual[14], "bb_width_weight": individual[15], "short_rsi_buy": individual[16], "short_rsi_sell": individual[17],
        "bb_width_low": individual[18], "bb_width_high": individual[19], "w_short_buy": individual[20], "w_short_sell": individual[21]
    }

    cash = 100000
    holdings = {'TSLA': 0, 'TSLL': 0}
    current_tsll_weight = 0.0
    prev_target_tsll_weight = -1.0
    portfolio_values = []

    for i in range(len(fold_data)):
        row = fold_data.iloc[i]
        fear_greed = row['y']
        daily_rsi = row['RSI_TSLA']
        weekly_rsi = row['Weekly RSI_TSLA']
        daily_rsi_trend = "Stable" if i < 10 else linregress(range(10), fold_data['RSI_TSLA'].iloc[i-10:i])[0] > 0 and "Increasing" or "Decreasing"
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
        obv_prev = fold_data['OBV_TSLA'].iloc[i-1] if i > 0 else obv
        bb_width = row['BB_width_TSLA']
        rsi5 = row['RSI5_TSLA']
        macd_short = row['MACD_short_TSLA']
        macd_signal_short = row['MACD_signal_short_TSLA']
        vwap = row['VWAP_TSLA']

        target_tsll_weight, _ = get_target_tsll_weight(
            fear_greed, daily_rsi, weekly_rsi, daily_rsi_trend, close, sma5, sma10, sma50, sma200, macd, macd_signal, macd_histogram,
            volume_change, atr, lower_band, upper_band, stochastic_k, stochastic_d, obv, obv_prev, bb_width, rsi5, macd_short, macd_signal_short, vwap,
            current_tsll_weight, params, row.name.strftime('%Y-%m-%d')
        )
        target_tsll_weight = min(target_tsll_weight, MAX_POSITION_SIZE)  # 최대 포지션 크기 제한
        target_tsla_weight = 1 - target_tsll_weight

        if abs(target_tsll_weight - prev_target_tsll_weight) > 0.01:
            total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
            if total_value / 100000 - 1 < STOP_LOSS_THRESHOLD:  # 손절매
                cash += holdings['TSLA'] * row['Close_TSLA'] * (1 - TRANSACTION_COST) + holdings['TSLL'] * row['Close_TSLL'] * (1 - TRANSACTION_COST)
                holdings['TSLA'], holdings['TSLL'] = 0, 0
            else:
                required_tsll_shares = int((target_tsll_weight * total_value) / row['Close_TSLL'])
                required_tsla_shares = int((target_tsla_weight * total_value) / row['Close_TSLA'])

                if required_tsll_shares < holdings['TSLL']:
                    sell_shares = holdings['TSLL'] - required_tsll_shares
                    cash += sell_shares * row['Close_TSLL'] * (1 - TRANSACTION_COST)
                    holdings['TSLL'] -= sell_shares
                elif required_tsll_shares > holdings['TSLL']:
                    buy_shares = required_tsll_shares - holdings['TSLL']
                    cost = buy_shares * row['Close_TSLL'] * (1 + TRANSACTION_COST)
                    if cash >= cost:
                        cash -= cost
                        holdings['TSLL'] += buy_shares

                if required_tsla_shares < holdings['TSLA']:
                    sell_shares = holdings['TSLA'] - required_tsla_shares
                    cash += sell_shares * row['Close_TSLA'] * (1 - TRANSACTION_COST)
                    holdings['TSLA'] -= sell_shares
                elif required_tsla_shares > holdings['TSLA']:
                    buy_shares = required_tsla_shares - holdings['TSLA']
                    cost = buy_shares * row['Close_TSLA'] * (1 + TRANSACTION_COST)
                    if cash >= cost:
                        cash -= cost
                        holdings['TSLA'] += buy_shares

            prev_target_tsll_weight = target_tsll_weight

        total_value = holdings['TSLA'] * row['Close_TSLA'] + holdings['TSLL'] * row['Close_TSLL'] + cash
        current_tsll_weight = (holdings['TSLL'] * row['Close_TSLL']) / total_value if total_value > 0 else 0
        portfolio_values.append(total_value)

    return calculate_fitness(np.array(portfolio_values)),

def optimize_parameters():
    """파라미터 최적화"""
    init_process()
    param_ranges = get_dynamic_param_ranges(volatility)

    # DEAP 설정
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda r: random.uniform(r[0], r[1]))
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [lambda: toolbox.attr_float(r) for r in param_ranges], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 교차 검증 설정
    tscv = TimeSeriesSplit(n_splits=5)
    fitness_scores = []

    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        def eval_with_fold(individual):
            return evaluate_individual(individual, test_data)

        toolbox.register("evaluate", eval_with_fold)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=POPULATION_SIZE)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        best_fitness = -np.inf
        patience_counter = 0

        for gen in tqdm(range(NUM_GENERATIONS), desc="Generation"):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population = toolbox.select(offspring, k=len(population))
            top_fitness = max([ind.fitness.values[0] for ind in population])
            if top_fitness > best_fitness:
                best_fitness = top_fitness
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        best_ind = tools.selBest(population, k=1)[0]
        fitness_scores.append(best_ind.fitness.values[0])

    # 최종 최적 파라미터
    best_population = toolbox.population(n=POPULATION_SIZE)
    fits = list(map(lambda ind: evaluate_individual(ind, data), best_population))
    for fit, ind in zip(fits, best_population):
        ind.fitness.values = fit
    best_individual = tools.selBest(best_population, k=1)[0]

    param_names = ["fg_buy", "fg_sell", "daily_rsi_buy", "daily_rsi_sell", "weekly_rsi_buy", "weekly_rsi_sell",
                   "volume_change_strong_buy", "volume_change_weak_buy", "volume_change_sell", "w_strong_buy",
                   "w_weak_buy", "w_sell", "stochastic_buy", "stochastic_sell", "obv_weight", "bb_width_weight",
                   "short_rsi_buy", "short_rsi_sell", "bb_width_low", "bb_width_high", "w_short_buy", "w_short_sell"]
    optimal_params = dict(zip(param_names, best_individual))

    with open("optimal_params.json", "w") as f:
        json.dump({"version": "2.1", "parameters": optimal_params}, f, indent=4)

    print("Optimal Parameters:", optimal_params)
    print("Maximum Fitness (Weighted Sharpe, Calmar, Total Return, Sortino):", best_individual.fitness.values[0])

if __name__ == "__main__":
    optimize_parameters()
