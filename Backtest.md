# Portfolio Backtesting with Genetic Algorithm (`backtest.py`)

## Overview

`backtest.py`는 유전 알고리즘(Genetic Algorithm, GA)을 활용하여 포트폴리오 조정 전략의 최적 파라미터를 도출하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터(TSLA 및 TSLL), 공포탐욕지수(Fear & Greed Index), VIX 변동성 지수를 기반으로 백테스트를 수행하여 포트폴리오의 **위험 조정 수익률**(Calmar Ratio 기반)을 극대화하는 파라미터를 찾습니다. 최적화된 파라미터는 `optimal_params.json` 파일로 저장됩니다.

---

## Features

- **유전 알고리즘(GA)**: DEAP 라이브러리를 사용하여 최적 파라미터를 탐색합니다.
- **진행률 표시**: `tqdm`을 통해 백테스트와 GA의 진행 상황을 시각적으로 확인할 수 있습니다.
- **기술적 지표**: RSI, SMA(50, 200), MACD, 볼린저 밴드, ATR, Stochastic Oscillator, OBV, BB Width, MACD Histogram, 주간 RSI 등 다양한 지표를 활용합니다.
- **동적 파라미터 범위**: VIX 변동성 수준에 따라 파라미터 범위를 동적으로 조정합니다.
- **위험 조정 수익률**: 샤프 비율과 최대 손실(Max Drawdown)을 고려한 Calmar Ratio를 피트니스 함수로 사용하여 안정적인 전략을 선호합니다.
- **병렬 처리**: `multiprocessing`을 사용하여 백테스트 속도를 향상시킵니다.
- **조기 종료**: 일정 세대 동안 개선이 없을 경우 GA를 조기에 종료하여 불필요한 계산을 줄입니다.
- **교차 검증**: 데이터를 여러 폴드로 나누어 안정적인 평가를 수행합니다.

---

## Requirements

- **Python**: 3.6 이상
- **필요한 라이브러리**:
  - `pandas`
  - `numpy`
  - `deap`
  - `scipy`
  - `tqdm`
  - `multiprocessing`

### Installation

1. **필수 라이브러리 설치**:
   ```bash
   pip install pandas numpy deap scipy tqdm
   ```

2. **데이터 파일 준비**:
   스크립트와 동일한 디렉토리에 다음 CSV 파일을 준비하세요.
   - `fear_greed_2years.csv`: 공포탐욕지수 데이터 (컬럼: `date`, `y`, 날짜 형식: `YYYY-MM-DD`)
   - `TSLA-history-2y.csv`: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `TSLL-history-2y.csv`: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `VIX-history-2y.csv`: VIX 지수 데이터 (컬럼: `Date`, `VIX`, 날짜 형식: `MM/DD/YYYY`)

   **참고**: 모든 파일은 CSV 형식이어야 하며, 날짜 형식이 스크립트와 일치해야 합니다. 데이터를 수집하려면 다음 명령어를 실행하세요.
   ```bash
   python3 collect_market_data.py
   ```

---

## Usage

### 스크립트 실행

터미널에서 다음 명령어로 스크립트를 실행하세요.
```bash
python3 backtest.py
```

- **진행 상황**: `tqdm`을 통해 백테스트와 GA의 진행률이 표시됩니다.
- **출력**: 최적 파라미터와 해당 파라미터로 얻은 최대 Calmar Ratio가 콘솔에 출력되며, `optimal_params.json` 파일로 저장됩니다.

### 예제 출력

```
유전 알고리즘 진행: 100%|██████████| 100/100 [05:00<00:00,  3.00s/it]
최적 파라미터: {
    "fg_buy": 28.01, "fg_sell": 73.43, "daily_rsi_buy": 38.66, "daily_rsi_sell": 82.27,
    "weekly_rsi_buy": 37.59, "weekly_rsi_sell": 72.84, "volume_change_strong_buy": 0.38,
    "volume_change_weak_buy": 0.46, "volume_change_sell": -0.84,
    "w_strong_buy": 2.96, "w_weak_buy": 1.49, "w_sell": 1.70,
    "stochastic_buy": 25.12, "stochastic_sell": 75.34,
    "obv_weight": 1.23, "bb_width_weight": 1.45
}
최대 피트니스 (Calmar Ratio 기반): 0.2255
```

---

## Code Structure

### 주요 함수

1. **`init_process()`**:
   - **설명**: TSLA, TSLL, 공포탐욕지수, VIX 데이터를 로드하고 병합합니다. RSI, SMA, MACD, 볼린저 밴드, ATR, Stochastic Oscillator, OBV, BB Width, MACD Histogram, 주간 RSI 등 기술적 지표를 계산하여 데이터프레임에 추가합니다.
   - **입력**: 없음 (CSV 파일에서 데이터 로드).
   - **출력**: 전처리 및 지표 계산이 완료된 데이터프레임.

2. **지표 계산 함수**:
   - **`calculate_rsi(series, timeperiod)`**: 상대강도지수(RSI) 계산.
   - **`calculate_sma(series, timeperiod)`**: 단순이동평균(SMA) 계산.
   - **`calculate_macd(series, fastperiod, slowperiod, signalperiod)`**: MACD와 시그널 라인 계산.
   - **`calculate_bollinger_bands(series, timeperiod, nbdevup, nbdevdn)`**: 볼린저 밴드(상단, 중간, 하단) 계산.
   - **`calculate_atr(df, timeperiod)`**: 평균 진폭(ATR) 계산.
   - **`calculate_stochastic(df, k_period, d_period)`**: Stochastic Oscillator %K와 %D 계산.
   - **`calculate_obv(close, volume)`**: 온밸런스 볼륨(OBV) 계산.

3. **`get_dynamic_param_ranges(volatility)`**:
   - **설명**: VIX 변동성에 따라 파라미터 범위를 동적으로 설정합니다.
   - **로직**:
     - 높은 변동성 (VIX > 30): 더 넓은 범위 (예: `fg_buy`: 15–50, `fg_sell`: 50–90).
     - 낮은 변동성: 더 좁은 범위 (예: `fg_buy`: 20–40, `fg_sell`: 60–85).
   - **출력**: 파라미터 범위를 정의한 튜플 리스트.

4. **`evaluate(individual, data_subset=None)`**:
   - **설명**: 주어진 파라미터로 백테스트를 수행하고 Calmar Ratio를 계산합니다. 교차 검증을 통해 안정성을 높입니다.
   - **로직**: 일일 포트폴리오 조정을 시뮬레이션하고, 일일 수익률을 계산하여 Calmar Ratio를 `(샤프 비율 / (1 + 최대 손실))`로 계산.
   - **입력**: GA 개체(파라미터 리스트), 데이터프레임.
   - **출력**: Calmar Ratio 기반 피트니스 스코어.

5. **`get_target_tsll_weight(...)`**:
   - **설명**: 다양한 지표를 기반으로 TSLL의 목표 비중을 계산합니다. ATR을 사용하여 매수/매도 조건을 동적으로 조정.
   - **입력**: 현재 시장 데이터 및 파라미터.
   - **출력**: TSLL의 목표 비중.

6. **`main()`**:
   - **설명**: GA를 설정하고 실행하여 최적 파라미터를 도출합니다. 병렬 처리와 조기 종료를 적용하여 효율성을 높임. 결과를 출력하고 `optimal_params.json`에 저장.
   - **입력**: 없음.
   - **출력**: 최적 파라미터와 최대 Calmar Ratio.

### Logic Explanation

- **유전 알고리즘 설정**:
  - **개체군**: 파라미터 집합(개체)으로 구성.
  - **적합도 함수**: `evaluate`에서 Calmar Ratio를 계산.
  - **교차**: `cxBlend`를 사용하여 두 개체의 파라미터를 블렌딩.
  - **돌연변이**: `mutGaussian`을 사용하여 파라미터에 가우시안 노이즈 추가.
  - **선택**: `selTournament`를 통해 토너먼트 방식으로 우수한 개체 선택.

- **백테스트 로직**:
  1. **초기화**: $100,000, TSLA와 TSLL 주식 0주로 시작.
  2. **일일 조정**:
     - `get_target_tsll_weight`를 통해 TSLL 목표 비중을 계산.
     - 목표 비중에 맞춰 TSLA와 TSLL을 매수/매도.
  3. **성능 지표**: 일일 포트폴리오 가치, 수익률, Calmar Ratio를 계산.

- **매수/매도 조건**:
  - **매수**:
    - 공포탐욕지수 ≤ `fg_buy`
    - 일일 RSI < `daily_rsi_buy`
    - MACD > 시그널 (음수 구간)
    - 거래량 변화율 > `volume_change_strong_buy` 또는 `volume_change_weak_buy` (ATR 조정)
    - 주가 < 하단 볼린저 밴드
    - RSI 상승 & 주가 > SMA200
    - Stochastic %K < `stochastic_buy`
    - OBV 증가
    - BB Width < 0.05
    - MACD Histogram > 0
  - **매도**:
    - 공포탐욕지수 ≥ `fg_sell`
    - 일일 RSI > `daily_rsi_sell`
    - MACD < 시그널 (양수 구간)
    - 거래량 변화율 < `volume_change_sell` (ATR 조정)
    - 주가 > 상단 볼린저 밴드
    - RSI 하락 & 주가 < SMA200
    - Stochastic %K > `stochastic_sell`
    - OBV 감소
    - BB Width > 0.15
    - MACD Histogram < 0

---

## Genetic Algorithm Parameters

### 개체군 크기와 세대 수

GA의 성능은 **개체군 크기**와 **세대 수**에 크게 의존하며, 최적화 품질과 계산 비용 간의 균형을 맞추는 데 중요합니다.

- **개체군 크기**:
  - **정의**: 한 세대에서 평가되는 파라미터 집합의 수.
  - **영향**:
    - 작을 경우 (20–50): 계산이 빠르지만 지역 최적해에 갇힐 가능성 높음.
    - 클 경우 (200 이상): 전역 최적해 탐색 가능성 높아지지만 계산 부담 증가.
  - **권장**: 파라미터 수(16개)의 5–10배, 즉 80–160. 기본값: 200.

- **세대 수**:
  - **정의**: 진화 과정을 반복하는 횟수.
  - **영향**:
    - 적을 경우 (20–30): 충분한 최적화가 이루어지지 않음.
    - 많을 경우 (200 이상): 최적해에 근접할 가능성 높지만 과적합 위험.
  - **권장**: 50–100. 기본값: 100.

- **최적 조합**:
  - 평가 횟수(개체군 × 세대)가 5,000–10,000번을 목표로 설정. 예: 200 × 100 = 20,000.
  - 자원에 따라 조정 가능; 병렬 처리를 통해 실행 시간 단축.

### Other GA Parameters

- **교차 확률**: 0.8
- **돌연변이 확률**: 0.4
- **토너먼트 크기**: 3
- **조기 종료 인내심**: 15세대 동안 개선 없을 시 종료
- **파라미터 클리핑**: 돌연변이/교차 후 동적 범위 내로 제한

---

## Parameter Ranges

| 파라미터                   | 고변동성 범위 | 저변동성 범위 | 설명                         |
|---------------------------|--------------|--------------|----------------------------|
| `fg_buy`                  | 15~50        | 20~40        | 매수 시 공포탐욕지수 하한     |
| `fg_sell`                 | 50~90        | 60~85        | 매도 시 공포탐욕지수 상한     |
| `daily_rsi_buy`           | 20~50        | 25~40        | 매수 시 일일 RSI 하한        |
| `daily_rsi_sell`          | 60~90        | 65~85        | 매도 시 일일 RSI 상한        |
| `weekly_rsi_buy`          | 25~50        | 30~40        | 매수 시 주간 RSI 하한        |
| `weekly_rsi_sell`         | 60~90        | 65~85        | 매도 시 주간 RSI 상한        |
| `volume_change_strong_buy` | 0.2~1.0      | 0.3~1.0      | 강한 매수 시 거래량 변화율    |
| `volume_change_weak_buy`  | 0.05~0.5     | 0.1~0.5      | 약한 매수 시 거래량 변화율    |
| `volume_change_sell`      | -1.0~-0.05   | -1.0~-0.1    | 매도 시 거래량 변화율        |
| `w_strong_buy`            | 1.0~3.0      | 1.5~3.0      | 강한 매수 비중 조정 가중치    |
| `w_weak_buy`              | 0.5~2.0      | 0.5~2.0      | 약한 매수 비중 조정 가중치    |
| `w_sell`                  | 0.5~2.0      | 0.5~2.0      | 매도 비중 조정 가중치        |
| `stochastic_buy`          | 20~40        | 20~40        | 매수 시 Stochastic %K 하한   |
| `stochastic_sell`         | 60~80        | 60~80        | 매도 시 Stochastic %K 상한   |
| `obv_weight`              | 0.5~2.0      | 0.5~2.0      | OBV 신호 가중치             |
| `bb_width_weight`         | 0.5~2.0      | 0.5~2.0      | BB Width 신호 가중치        |

참고 사항
고변동성 vs 저변동성: 고변동성 환경에서는 파라미터 범위가 더 넓어 전략이 시장 변화에 유연하게 대응할 수 있도록 설계되었습니다. 반면, 저변동성 환경에서는 범위가 좁아 안정적인 시장에 적합한 값을 탐색합니다.
추가된 파라미터: 사용자가 제공한 기존 표와 비교해 stochastic_buy, stochastic_sell, obv_weight, bb_width_weight가 추가되었습니다. 이는 개선된 코드에서 새로운 지표(Stochastic Oscillator, OBV, Bollinger Band Width)를 활용하기 때문입니다.

---

## Improvements

1. **동적 파라미터 범위**:
   - VIX를 사용해 시장 변동성에 따라 파라미터 범위를 조정하여 전략의 유연성을 높임.

2. **위험 조정 수익률**:
   - Calmar Ratio를 사용하여 샤프 비율과 최대 손실을 동시에 고려한 안정적인 전략을 선호.

3. **성능 향상**:
   - **병렬 처리**: `multiprocessing`을 사용해 최대 4코어로 속도 향상.
   - **조기 종료**: 15세대 동안 개선이 없을 시 GA를 종료하여 계산 자원 절약.
   - **교차 검증**: 5폴드 교차 검증을 통해 과적합을 방지하고 안정적인 결과 도출.

4. **추가 지표**:
   - Stochastic Oscillator, OBV, BB Width, MACD Histogram 등 새로운 지표를 추가하여 매수/매도 조건 강화.

---

## Disclaimer

이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
