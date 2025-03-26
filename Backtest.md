# Portfolio Backtesting with Genetic Algorithm (`backtest.py`)

## Overview

`backtest.py`는 유전 알고리즘(Genetic Algorithm, GA)을 활용하여 포트폴리오 조정 전략의 최적 파라미터를 도출하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터(TSLA 및 TSLL), 공포탐욕지수(Fear & Greed Index), VIX 변동성 지수를 기반으로 백테스트를 수행하여 포트폴리오의 **위험 조정 수익률**(샤프 비율에 총 수익률을 가중한 값)을 극대화하는 파라미터를 찾습니다. 최적화된 파라미터는 `optimal_params.json` 파일로 저장됩니다.

## Features

- **유전 알고리즘(GA)**: DEAP 라이브러리를 사용하여 최적 파라미터를 탐색합니다.
- **진행률 표시**: `tqdm`을 통해 백테스트와 GA의 진행 상황을 시각적으로 확인할 수 있습니다.
- **기술적 지표**: RSI, SMA(50, 200), MACD, 볼린저 밴드, ATR, 주간 RSI 등 다양한 지표를 활용합니다.
- **동적 파라미터 범위**: VIX 변동성 수준에 따라 파라미터 범위를 동적으로 조정합니다.
- **위험 조정 수익률**: 샤프 비율을 기반으로 총 수익률을 가중하여 전략을 평가합니다.
- **병렬 처리**: `multiprocessing`을 사용하여 백테스트 속도를 향상시킵니다.
- **조기 종료**: 일정 세대 동안 개선이 없을 경우 GA를 조기에 종료하여 불필요한 계산을 줄입니다.

## Requirements

- **Python**: 3.6 이상
- **필요한 라이브러리**:
  - `pandas`
  - `numpy`
  - `deap`
  - `scipy`
  - `tqdm`
  - `yfinance` (선택사항, VIX 데이터 수집용)

### Installation

1. **필수 라이브러리 설치**:
   ```bash
   pip install pandas numpy deap scipy tqdm yfinance
   ```

2. **데이터 파일 준비**:
   스크립트와 동일한 디렉토리에 다음 CSV 파일을 준비하세요.
   ```bash
   python3 collect_market_data.py
   ```
   - `fear_greed_2years.csv`: 공포탐욕지수 데이터 (컬럼: `date`, `y`, 날짜 형식: `YYYY-MM-DD`)
   - `TSLA-history-2y.csv`: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `TSLL-history-2y.csv`: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `VIX-history-2y.csv`: VIX 지수 데이터 (컬럼: `Date`, `VIX`, 날짜 형식: `MM/DD/YYYY`)

   **참고**: 모든 파일은 CSV 형식이어야 하며, 날짜 형식이 스크립트와 일치해야 합니다.

## Usage

### 스크립트 실행

터미널에서 다음 명령어로 스크립트를 실행하세요:
```bash
python3 backtest.py
```

- **진행 상황**: `tqdm`을 통해 백테스트와 GA의 진행률이 표시됩니다.
- **출력**: 최적 파라미터와 해당 파라미터로 얻은 최대 가중 샤프 비율이 콘솔에 출력되며, `optimal_params.json` 파일로 저장됩니다.

### 예제 출력

```
Genetic Algorithm Progress: 100%|██████████| 50/50 [02:30<00:00,  3.00s/it]
최적 파라미터: {
    "fg_buy": 28.01, "fg_sell": 73.43, "daily_rsi_buy": 38.66, "daily_rsi_sell": 82.27,
    "weekly_rsi_buy": 37.59, "weekly_rsi_sell": 72.84, "volume_change_strong_buy": 0.38,
    "volume_change_weak_buy": 0.46, "volume_change_sell": -0.84,
    "w_strong_buy": 2.96, "w_weak_buy": 1.49, "w_sell": 1.70
}
최대 가중치 샤프 비율: 0.2255
```

## Code Structure

### 함수

1. **`init_process()`**:
   - **설명**: TSLA, TSLL, 공포탐욕지수, VIX 데이터를 로드하고 병합합니다. RSI, SMA, MACD, 볼린저 밴드, ATR, 주간 RSI 등 기술적 지표를 계산하여 데이터프레임에 추가합니다.
   - **입력**: 없음 (CSV 파일에서 데이터 로드).
   - **출력**: 전처리 및 지표 계산이 완료된 데이터프레임.

2. **지표 계산 함수**:
   - **`calculate_rsi(series, timeperiod)`**: 상대강도지수(RSI) 계산.
   - **`calculate_sma(series, timeperiod)`**: 단순이동평균(SMA) 계산.
   - **`calculate_macd(series, fastperiod, slowperiod, signalperiod)`**: MACD와 시그널 라인 계산.
   - **`calculate_bollinger_bands(series, timeperiod, nbdevup, nbdevdn)`**: 볼린저 밴드(상단, 중간, 하단) 계산.
   - **`calculate_atr(df, timeperiod)`**: 평균 진폭(ATR) 계산.
   - **`get_rsi_trend(rsi_series, window)`**: RSI 추세(상승, 하락, 안정) 판단.

3. **`get_dynamic_param_ranges(volatility)`**:
   - **설명**: VIX 변동성에 따라 파라미터 범위를 동적으로 설정합니다.
   - **로직**:
     - 높은 변동성 (VIX > 30): 더 넓은 범위 (예: `fg_buy`: 15–50, `fg_sell`: 50–90).
     - 낮은 변동성: 더 좁은 범위 (예: `fg_buy`: 20–40, `fg_sell`: 60–85).
   - **출력**: 파라미터 범위를 정의한 튜플 리스트.

4. **`evaluate(individual)`**:
   - **설명**: 주어진 파라미터로 백테스트를 수행하고 가중 샤프 비율을 계산합니다.
   - **로직**: 일일 포트폴리오 조정을 시뮬레이션하고, 일일 수익률을 계산하여 샤프 비율을 `(평균 수익률 / 표준편차) * (1 + 총 수익률)`로 계산.
   - **입력**: GA 개체(파라미터 리스트), 데이터프레임.
   - **출력**: 가중 샤프 비율.

5. **`get_target_tsll_weight(...)`**:
   - **설명**: 공포탐욕지수, RSI, MACD, 볼린저 밴드, 거래량 변화율 등 다양한 지표를 기반으로 TSLL의 목표 비중을 계산합니다. ATR을 사용하여 매수/매도 조건을 동적으로 조정.
   - **입력**: 현재 시장 데이터 및 파라미터.
   - **출력**: TSLL의 목표 비중.

6. **`main()`**:
   - **설명**: GA를 설정하고 실행하여 최적 파라미터를 도출합니다. 병렬 처리와 조기 종료를 적용하여 효율성을 높임. 결과를 출력하고 `optimal_params.json`에 저장.
   - **입력**: 없음.
   - **출력**: 최적 파라미터와 최대 가중 샤프 비율.

### Logic Explanation

- **유전 알고리즘 설정**:
  - **개체군**: 파라미터 집합(개체)으로 구성.
  - **적합도 함수**: `evaluate`에서 가중 샤프 비율을 계산.
  - **교차**: `cxBlend`를 사용하여 두 개체의 파라미터를 블렌딩.
  - **돌연변이**: `mutGaussian`을 사용하여 파라미터에 가우시안 노이즈 추가.
  - **선택**: `selTournament`를 통해 토너먼트 방식으로 우수한 개체 선택.

- **백테스트 로직**:
  1. **초기화**: $100,000, TSLA와 TSLL 주식 0주로 시작.
  2. **일일 조정**:
     - `get_target_tsll_weight`를 통해 TSLL 목표 비중을 계산.
     - 목표 비중에 맞춰 TSLA와 TSLL을 매수/매도.
  3. **성능 지표**: 일일 포트폴리오 가치, 수익률, 가중 샤프 비율을 계산.

- **매수/매도 조건**:
  - **매수**:
    - 공포탐욕지수 ≤ `fg_buy`
    - 일일 RSI < `daily_rsi_buy`
    - MACD > 시그널 (음수 구간)
    - 거래량 변화율 > `volume_change_strong_buy` 또는 `volume_change_weak_buy` (ATR 조정)
    - 주가 < 하단 볼린저 밴드
    - RSI 상승 & 주가 > SMA200
  - **매도**:
    - 공포탐욕지수 ≥ `fg_sell`
    - 일일 RSI > `daily_rsi_sell`
    - MACD < 시그널 (양수 구간)
    - 거래량 변화율 < `volume_change_sell` (ATR 조정)
    - 주가 > 상단 볼린저 밴드
    - RSI 하락 & 주가 < SMA200

## Genetic Algorithm Parameters

### 개체군 크기와 세대 수

GA의 성능은 **개체군 크기**와 **세대 수**에 크게 의존하며, 최적화 품질과 계산 비용 간의 균형을 맞추는 데 중요합니다.

- **개체군 크기**:
  - **정의**: 한 세대에서 평가되는 파라미터 집합의 수.
  - **영향**:
    - 작을 경우 (20–50): 계산이 빠르지만 지역 최적해에 갇힐 가능성 높음.
    - 클 경우 (200 이상): 전역 최적해 탐색 가능성 높아지지만 계산 부담 증가.
  - **권장**: 파라미터 수(12개)의 5–10배, 즉 60–120. 기본값: 100.

- **세대 수**:
  - **정의**: 진화 과정을 반복하는 횟수.
  - **영향**:
    - 적을 경우 (20–30): 충분한 최적화가 이루어지지 않음.
    - 많을 경우 (200 이상): 최적해에 근접할 가능성 높지만 과적합 위험.
  - **권장**: 50–100. 기본값: 50.

- **최적 조합**:
  - 평가 횟수(개체군 × 세대)가 5,000–10,000번을 목표로 설정. 예: 100 × 50 = 5,000.
  - 자원에 따라 조정 가능; 병렬 처리를 통해 실행 시간 단축.

### Other GA parameters

- **교차 확률**: 0.7
- **돌연변이 확률**: 0.3
- **토너먼트 크기**: 3
- **조기 종료 인내심**: 10세대 동안 개선 없을 시 종료
- **파라미터 클리핑**: 돌연변이/교차 후 동적 범위 내로 제한

## Improvements

1. **동적 파라미터 범위**:
   - VIX를 사용해 시장 변동성에 따라 파라미터 범위를 조정하여 유연성을 높임.

2. **위험 조정 수익률**:
   - 단순 수익률 대신 샤프 비율에 총 수익률을 가중한 `(샤프 비율 * (1 + 총 수익률))`을 사용하여 안정적이고 수익성 있는 전략을 선호.

3. **성능 향상**:
   - **병렬 처리**: `multiprocessing`을 사용해 최대 4코어로 속도 향상.
   - **조기 종료**: 10세대 동안 개선이 없을 시 GA를 종료하여 계산 자원 절약.

## Disclaimer

이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
