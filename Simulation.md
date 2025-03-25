# Portfolio Simulation (`simulation.py`)

## Overview

`simulation.py`는 지정된 기간 동안 TSLA와 TSLL 주식으로 구성된 포트폴리오를 시뮬레이션하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터와 공포탐욕지수(Fear & Greed Index)를 활용하여 포트폴리오의 성과를 분석합니다. 시뮬레이션 결과는 최종 포트폴리오 가치, 수익률, 주식 보유량, 현금 잔액 등을 포함한 요약 테이블로 출력됩니다.

## Features

- **명령줄 인자 지원**: `--start_date`와 `--days`를 통해 시뮬레이션 시작 날짜와 기간을 유연하게 설정 가능.
- **기술적 지표 계산**: RSI, SMA, MACD, 볼린저 밴드, ATR 등 다양한 지표를 계산하여 포트폴리오 조정에 활용.
- **현금 관리**: 포트폴리오 내 현금 잔액을 추적하고, 현금 비중과 금액을 결과에 포함.
- **요약 테이블**: `tabulate` 라이브러리를 사용하여 결과를 깔끔한 테이블 형식으로 출력.

## Requirements

- **Python**: 3.6 이상
- **필요한 라이브러리**:
  - `pandas`
  - `numpy`
  - `scipy`
  - `tabulate`

### Installation

1. **필수 라이브러리 설치**
   ```bash
   pip install pandas numpy scipy tabulate
   ```

2. **데이터 파일 준비**
   아래 파일들이 스크립트와 동일한 디렉토리에 있어야 합니다:
   - `fear_greed_2years.csv`: 공포탐욕지수 데이터 (컬럼: `date`, `y`)
   - `TSLA-history-2y.csv`: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume` 등)
   - `TSLL-history-2y.csv`: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume` 등)
   - `optimal_params.json`: 최적화된 파라미터 파일 (선택 사항, 없으면 기본값 사용)

   **참고**: 데이터 파일은 CSV 형식이며, 날짜 형식이 코드와 일치해야 합니다 (`TSLA`와 `TSLL`은 `MM/DD/YYYY`, 공포탐욕지수는 `YYYY-MM-DD`).

## Usage

### Command Line Arguments

- `--start_date`: 시뮬레이션 시작 날짜 (형식: `YYYY-MM-DD`). 생략 시 현재 날짜를 기준으로 과거 데이터를 사용.
- `--days`: 시뮬레이션 기간 (일 수). 생략 시 기본값 180일 사용.

### Examples

1. **특정 날짜부터 시뮬레이션**:
   ```bash
   python simulation.py --start_date 2024-01-01 --days 180
   ```
   - 2024년 1월 1일부터 180일 동안 시뮬레이션 실행.

2. **현재 날짜 기준 과거 시뮬레이션**:
   ```bash
   python simulation.py --days 180
   ```
   - 현재 날짜에서 180일 전부터 현재까지 시뮬레이션 실행.

### Output

시뮬레이션 결과는 다음과 같은 형식으로 터미널에 출력됩니다:

```
### Portfolio Simulation Results (180 days)
- Simulation Period: 2024-01-01 to 2024-06-29
- Initial Portfolio Value: $100000.00
- Final Portfolio Value: $81189.00
- Portfolio Returns: -18.81%

### Current Stock Price (2024-06-29)
- **TSLA Close**: $278.39
- **TSLL Close**: $11.16

### Summary Table
╒═════════════════╤════════════╤════════════╕
│ Metric          │ Start      │ End        │
╞═════════════════╪════════════╪════════════╡
│ Date            │ 2024-01-01 │ 2024-06-29 │
├─────────────────┼────────────┼────────────┤
│ Portfolio Value │ $100000.00 │ $81189.00  │
├─────────────────┼────────────┼────────────┤
│ TSLL Weight     │ 0.00%      │ 99.99%     │
├─────────────────┼────────────┼────────────┤
│ TSLA Weight     │ 0.00%      │ 0.00%      │
├─────────────────┼────────────┼────────────┤
│ Cash Weight     │ 100.00%    │ 0.01%      │
├─────────────────┼────────────┼────────────┤
│ Cash Amount     │ $100000.00 │ $30.00     │
├─────────────────┼────────────┼────────────┤
│ TSLA Shares     │ 0          │ 0          │
├─────────────────┼────────────┼────────────┤
│ TSLL Shares     │ 0          │ 7275       │
╘═════════════════╧════════════╧════════════╛
```

## Code Structure

### Functions

1. **`load_data(start_date, end_date)`**:
   - 지정된 기간의 주가 데이터(TSLA, TSLL)와 공포탐욕지수를 로드하고 병합.
   - 기술적 지표(RSI, SMA50, SMA200, MACD, 볼린저 밴드, ATR, 주간 RSI)를 계산하여 데이터프레임에 추가.

2. **지표 계산 함수**:
   - `calculate_rsi(series, timeperiod)`: 상대강도지수(RSI) 계산.
   - `calculate_sma(series, timeperiod)`: 단순이동평균(SMA) 계산.
   - `calculate_macd(series, fastperiod, slowperiod, signalperiod)`: MACD와 시그널 라인 계산.
   - `calculate_bollinger_bands(series, timeperiod, nbdevup, nbdevdn)`: 볼린저 밴드 계산.
   - `calculate_atr(df, timeperiod)`: 평균 истин 범위(ATR) 계산.
   - `get_rsi_trend(rsi_series, window)`: RSI의 추세(상승/하락/안정)를 판단.

3. **`load_params(file_path="optimal_params.json")`**:
   - `optimal_params.json`에서 최적화된 매수/매도 파라미터를 로드. 파일이 없으면 기본값 사용:
     ```json
     {
         "fg_buy": 25, "fg_sell": 75, "daily_rsi_buy": 30, "daily_rsi_sell": 70,
         "weekly_rsi_buy": 40, "weekly_rsi_sell": 60, "volume_change_strong_buy": 0.5,
         "volume_change_weak_buy": 0.2, "volume_change_sell": -0.2,
         "w_strong_buy": 2.0, "w_weak_buy": 1.0, "w_sell": 1.0
     }
     ```

4. **`get_target_tsll_weight(...)`**:
   - 공포탐욕지수, RSI, MACD, 볼린저 밴드, 거래량 변화 등 다양한 지표를 기반으로 TSLL의 목표 비중을 계산.
   - 매수/매도 조건을 평가하여 비중을 조정.

5. **`simulate_portfolio(start_date, end_date, params)`**:
   - 지정된 기간 동안 포트폴리오를 시뮬레이션.
   - 매일 데이터를 기반으로 TSLL과 TSLA의 비중을 조정하고 현금을 관리.
   - 최종 포트폴리오 가치, 주식 보유량, 현금 잔액 등을 반환.

6. **`main()`**:
   - 명령줄 인자를 파싱하고, 시뮬레이션을 실행한 후 결과를 출력.

### Logic Explanation

- **포트폴리오 초기화**:
  - 초기 자산: $100,000 (현금).
  - TSLA와 TSLL 주식 보유량: 0주.

- **일일 조정**:
  1. **목표 비중 계산**: `get_target_tsll_weight` 함수를 통해 TSLL의 목표 비중을 계산하고, TSLA 비중은 `1 - TSLL 비중`으로 설정.
  2. **주식 매수/매도**:
     - 목표 주식 수 계산: `(목표 비중 × 총 포트폴리오 가치) / 주가`.
     - 현금이 충분하면 목표 주식 수만큼 매수/매도.
     - 현금이 부족하면 가능한 만큼만 매수.
  3. **TSLL 비중 100% 처리**: 목표 TSLL 비중이 100%일 경우, 남은 현금을 모두 TSLL에 투자. 단, 현금이 주식 한 주 가격보다 적으면 현금으로 유지.

- **최종 계산**:
  - 시뮬레이션 종료 후 최종 포트폴리오 가치 계산: `(TSLA 주식 수 × TSLA 종가) + (TSLL 주식 수 × TSLL 종가) + 현금`.
  - TSLL/TSLA 비중, 현금 비중 및 금액을 계산하여 요약 테이블로 출력.

- **매수/매도 조건**:
  - **매수 조건**:
    - 공포탐욕지수 ≤ 25
    - 일일 RSI < 30
    - MACD > 시그널 (음수 구간)
    - 거래량 변화율 > 0.5 (강한 매수) 또는 > 0.2 (약한 매수)
    - 주가 < 볼린저 하단 밴드
    - RSI 상승 추세 & 주가 > SMA200
  - **매도 조건**:
    - 공포탐욕지수 ≥ 75
    - 일일 RSI > 70
    - MACD < 시그널 (양수 구간)
    - 거래량 변화율 < -0.2
    - 주가 > 볼린저 상단 밴드
    - RSI 하락 추세 & 주가 < SMA200
