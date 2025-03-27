# Portfolio Backtesting with Genetic Algorithm (`backtest.py`)

## Overview

`backtest.py`는 유전 알고리즘(Genetic Algorithm, GA)을 활용하여 TSLA와 TSLL 주식으로 구성된 포트폴리오의 비중 조정 전략에 대한 최적 파라미터를 도출하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터, 공포탐욕지수(Fear & Greed Index), VIX 변동성 지수를 기반으로 백테스트를 수행하여 포트폴리오의 **위험 조정 수익률**(Calmar Ratio 기반)을 극대화하는 파라미터를 찾습니다. 최적화된 파라미터는 `optimal_params.json` 파일로 저장되며, 이는 실시간 포트폴리오 조정에 활용될 수 있습니다.

### Key Improvements

- **거래 비용 반영**: 매수 및 매도 시 0.1%의 거래 비용을 적용하여 현실적인 수익률을 평가합니다.
- **단기 지표 추가**: SMA5, SMA10, RSI5, 단기 MACD, VWAP 등 단기 예측 지표를 추가하여 After Market 반응성을 강화합니다.
- **동적 파라미터 범위**: VIX 변동성에 따라 파라미터 범위를 조정하여 시장 상황에 적응합니다.
- **향상된 피트니스 함수**: Calmar Ratio를 사용하여 수익률과 최대 손실을 동시에 고려합니다.

---

## Features

- **유전 알고리즘(GA)**: DEAP 라이브러리를 활용하여 파라미터 최적화를 수행합니다.
- **진행률 표시**: `tqdm`을 통해 GA 진행 상황을 시각적으로 확인할 수 있습니다.
- **기술적 지표**: RSI, SMA(5, 10, 50, 200), MACD, Bollinger Bands, ATR, Stochastic Oscillator, OBV, BB Width, VWAP, 단기 MACD 등 다양한 지표를 활용합니다.
- **동적 파라미터 범위**: VIX에 따라 파라미터 범위를 동적으로 설정합니다.
- **위험 조정 수익률**: Calmar Ratio를 피트니스 함수로 사용하여 안정적인 전략을 선호합니다.
- **병렬 처리**: `multiprocessing`을 사용하여 최대 4코어로 속도를 향상시킵니다.
- **조기 종료**: 일정 세대 동안 개선이 없을 시 GA를 조기에 종료합니다.
- **교차 검증**: 5폴드 교차 검증을 통해 안정적인 평가를 수행합니다.

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

1. **의존성 설치**:
   ```bash
   pip install pandas numpy deap scipy tqdm
   ```

2. **데이터 파일 준비**:
   스크립트와 동일한 디렉토리에 다음 파일이 필요합니다:
   - `fear_greed_2years.csv`: 공포탐욕지수 데이터 (컬럼: `date`, `y`, 날짜 형식: `YYYY-MM-DD`)
   - `TSLA-history-2y.csv`: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `TSLL-history-2y.csv`: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `VIX-history-2y.csv`: VIX 지수 데이터 (컬럼: `Date`, `VIX`, 날짜 형식: `MM/DD/YYYY`)

   **참고**: 데이터가 없는 경우 다음 명령어로 수집하세요:
   ```bash
   python3 collect_market_data.py
   ```

---

## Usage

### 스크립트 실행

터미널에서 다음 명령어로 실행합니다:
```bash
python3 backtest.py
```
- **진행 상황**: `tqdm`을 통해 GA 진행률이 표시됩니다.
- **출력**: 최적 파라미터와 Calmar Ratio가 콘솔에 출력되며, `optimal_params.json`에 저장됩니다.

### 예시 출력

```
유전 알고리즘 진행: 100%|██████████| 100/100 [05:00<00:00,  3.00s/it]
최적 파라미터: {
    "fg_buy": 28.01, "fg_sell": 73.43, "daily_rsi_buy": 38.66, "daily_rsi_sell": 82.27,
    "weekly_rsi_buy": 37.59, "weekly_rsi_sell": 72.84, "volume_change_strong_buy": 0.38,
    "volume_change_weak_buy": 0.46, "volume_change_sell": -0.84, "w_strong_buy": 2.96,
    "w_weak_buy": 1.49, "w_sell": 1.70, "stochastic_buy": 25.12, "stochastic_sell": 75.34,
    "obv_weight": 1.23, "bb_width_weight": 1.45, "short_rsi_buy": 25.12, "short_rsi_sell": 75.34,
    "bb_width_low": 0.1, "bb_width_high": 0.2, "w_short_buy": 1.5, "w_short_sell": 1.5
}
최대 피트니스 (Calmar Ratio 기반): 0.2255
```

---

## Code Structure

### Key Functions

1. **`init_process()`**:
   - **설명**: TSLA, TSLL, 공포탐욕지수 데이터를 로드하고 병합합니다. RSI, SMA(5, 10, 50, 200), MACD, Bollinger Bands, ATR, Stochastic Oscillator, OBV, VWAP, 단기 MACD 등 지표를 계산합니다.
   - **출력**: 전처리 및 지표 계산이 완료된 데이터프레임.

2. **`get_dynamic_param_ranges(volatility)`**:
   - **설명**: VIX 변동성에 따라 파라미터 범위를 동적으로 설정합니다.
   - **로직**: 고변동성(VIX > 30) 시 넓은 범위, 저변동성 시 좁은 범위.
   - **출력**: 파라미터 범위 튜플 리스트.

3. **`evaluate(individual, data_subset=None)`**:
   - **설명**: 주어진 파라미터로 백테스트를 수행하고 Calmar Ratio를 계산합니다. 교차 검증으로 안정성을 확보합니다.
   - **출력**: 피트니스 스코어(Calmar Ratio).

4. **`get_target_tsll_weight(...)`**:
   - **설명**: 다양한 지표를 기반으로 TSLL 목표 비중을 계산합니다. ATR을 활용해 동적 조정을 합니다.
   - **출력**: TSLL 목표 비중.

5. **`adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)`**:
   - **설명**: 목표 비중에 맞춰 포트폴리오를 조정하며, 0.1% 거래 비용을 반영합니다.
   - **출력**: 조정된 보유 주식과 현금.

6. **`main()`**:
   - **설명**: GA를 설정하고 실행하여 최적 파라미터를 도출하며, 결과를 `optimal_params.json`에 저장합니다.

### Logic Explanation

- **유전 알고리즘 설정**:
  - **개체군**: 파라미터 집합.
  - **적합도 함수**: Calmar Ratio.
  - **교차**: `cxBlend`.
  - **돌연변이**: `mutGaussian`.
  - **선택**: `selTournament`.

- **백테스트 로직**:
  - 초기 자산: $100,000.
  - 매일 목표 비중 계산 및 포트폴리오 조정.
  - 거래 비용 0.1% 적용.

- **매수/매도 조건**:
  - **매수**: 낮은 공포탐욕지수, 낮은 RSI, 양의 MACD 히스토그램, SMA5 > SMA10 등.
  - **매도**: 높은 공포탐욕지수, 높은 RSI, 음의 MACD 히스토그램, SMA5 < SMA10 등.

---

## Genetic Algorithm Parameters

- **개체군 크기**: 200
- **세대 수**: 100
- **교차 확률**: 0.8
- **돌연변이 확률**: 0.4
- **조기 종료**: 15세대 동안 개선 없으면 종료

---

## Parameter Ranges

| 파라미터                   | 고변동성 범위 | 저변동성 범위 | 설명                         |
|---------------------------|--------------|--------------|----------------------------|
| `fg_buy`                  | 15~50        | 20~40        | 매수 시 공포탐욕지수 하한     |
| `fg_sell`                 | 50~90        | 60~85        | 매도 시 공포탐욕지수 상한     |
| `daily_rsi_buy`           | 20~50        | 25~40        | 매수 시 일일 RSI 하한        |
| `daily_rsi_sell`          | 60~80        | 65~80        | 매도 시 일일 RSI 상한        |
| `weekly_rsi_buy`          | 25~50        | 30~40        | 매수 시 주간 RSI 하한        |
| `weekly_rsi_sell`         | 60~90        | 65~85        | 매도 시 주간 RSI 상한        |
| `volume_change_strong_buy`| 0.2~1.0      | 0.3~1.0      | 강한 매수 시 거래량 변화율    |
| `volume_change_weak_buy`  | 0.05~0.5     | 0.1~0.5      | 약한 매수 시 거래량 변화율    |
| `volume_change_sell`      | -0.5~-0.05   | -0.5~-0.1    | 매도 시 거래량 변화율        |
| `w_strong_buy`            | 1.0~3.0      | 1.5~3.0      | 강한 매수 가중치             |
| `w_weak_buy`              | 0.5~2.0      | 0.5~2.0      | 약한 매수 가중치             |
| `w_sell`                  | 0.5~2.0      | 0.5~2.0      | 매도 가중치                 |
| `stochastic_buy`          | 20~40        | 20~40        | 매수 시 Stochastic %K 하한   |
| `stochastic_sell`         | 70~80        | 70~80        | 매도 시 Stochastic %K 상한   |
| `obv_weight`              | 0.5~2.0      | 0.5~2.0      | OBV 신호 가중치             |
| `bb_width_weight`         | 0.5~2.0      | 0.5~2.0      | BB Width 신호 가중치        |
| `short_rsi_buy`           | 20~40        | 20~40        | 매수 시 단기 RSI 하한        |
| `short_rsi_sell`          | 60~80        | 60~80        | 매도 시 단기 RSI 상한        |
| `bb_width_low`            | 0.05~0.15    | 0.05~0.15    | BB 폭 낮음 기준             |
| `bb_width_high`           | 0.15~0.3     | 0.15~0.3     | BB 폭 높음 기준             |
| `w_short_buy`             | 0.5~2.0      | 0.5~2.0      | 단기 매수 가중치            |
| `w_short_sell`            | 0.5~2.0      | 0.5~2.0      | 단기 매도 가중치            |

---

## Improvements

- **거래 비용 반영**: 현실적인 수익률 평가를 위해 0.1% 거래 비용을 추가했습니다.
- **단기 지표 추가**: After Market 반응성을 강화하기 위해 SMA5, SMA10, RSI5, 단기 MACD, VWAP를 도입했습니다.
- **동적 파라미터 범위**: 시장 변동성에 따라 파라미터 범위를 조정하여 유연성을 확보했습니다.
- **향상된 피트니스 함수**: Calmar Ratio를 통해 수익률과 위험을 균형 있게 평가합니다.

---

## Disclaimer

이 프로그램은 교육 목적으로 제공되며, 투자 조언이 아닙니다. 투자 결정을 내리기 전 금융 전문가와 상담하세요.
