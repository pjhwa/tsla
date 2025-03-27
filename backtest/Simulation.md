# Portfolio Simulation (`simulation.py`)

## Overview
`simulation.py`는 TSLA와 TSLL 주식으로 구성된 포트폴리오의 성과를 지정된 기간 동안 시뮬레이션하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터, 공포탐욕지수(Fear & Greed Index), 그리고 최적화된 파라미터(`optimal_params.json`)를 활용하여 포트폴리오 비중을 동적으로 조정합니다. 시뮬레이션 결과는 최종 포트폴리오 가치, 수익률, 주식 보유량, 현금 잔액 등을 포함하며, 이를 요약 테이블로 출력합니다.

### Key Improvements
- **거래 비용 (Transaction Costs)**: 주식 매수 및 매도 시 0.1%의 거래 비용을 적용하여 실제 거래 조건을 반영.
- **After Market 조정**: 전일 데이터를 기반으로 포트폴리오 비중을 결정하고, 당일 종가로 거래를 실행.
- **향상된 파라미터 처리**: `optimal_params.json`에서 최적화된 파라미터를 로드하며, 파일이 없거나 최신 버전이 아니면 기본값을 사용.

---

## Features
- **명령줄 유연성**: `--start_date`와 `--days` 인자를 통해 시뮬레이션 기간을 동적으로 설정 가능.
- **기술적 지표**: RSI, SMA (5, 10, 50, 200), MACD, Bollinger Bands, ATR, Stochastic Oscillator, OBV, BB Width, VWAP 등 다양한 지표를 계산.
- **현금 관리**: 현금 잔액을 추적하고 거래 비용을 반영하여 현실적인 거래를 보장.
- **포트폴리오 재조정**: 시장 지표를 기반으로 매일 TSLA와 TSLL 보유량을 목표 비중에 맞춰 조정.
- **결과 요약**: `tabulate`를 사용하여 시뮬레이션 결과를 깔끔한 테이블로 출력(포트폴리오 가치, 수익률, 주식 비중, 현금 보유액 포함).

---

## Requirements
- **Python**: 3.6 이상
- **필요한 라이브러리**:
  - `pandas`
  - `numpy`
  - `scipy`
  - `tabulate`
  - `json`
  - `argparse`

### Installation
1. **의존성 설치**:
   ```bash
   pip install pandas numpy scipy tabulate
   ```

2. **데이터 파일**:
   스크립트와 동일한 디렉토리에 다음 파일이 필요합니다:
   - `fear_greed_2years.csv`: 공포탐욕지수 데이터 (컬럼: `date`, `y`)
   - `TSLA-history-2y.csv`: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume` 등)
   - `TSLL-history-2y.csv`: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume` 등)
   - `optimal_params.json`: 최적화된 파라미터 (선택 사항; 없으면 기본값 사용)

   **참고**: 데이터 파일이 없는 경우 다음 명령으로 수집 가능:
   ```bash
   python3 collect_market_data.py
   ```

---

## Usage

### Command-Line Arguments
- `--start_date`: 시뮬레이션 시작 날짜 (형식: `YYYY-MM-DD`). 생략 시 `--days`를 기준으로 과거 기간 설정.
- `--days`: 시뮬레이션 기간(일 수). `--start_date`와 함께 사용 시 기간 길이를 정의하며, 단독 사용 시 현재 날짜부터 과거로 계산.

### Examples
1. **특정 날짜부터 시뮬레이션**:
   ```bash
   python simulation.py --start_date 2024-01-01 --days 180
   ```
   - 2024년 1월 1일부터 180일 동안 시뮬레이션.

2. **과거 기간 시뮬레이션**:
   ```bash
   python simulation.py --days 180
   ```
   - 현재 날짜 기준 과거 180일 시뮬레이션.

### Output
시뮬레이션 결과는 터미널에 다음 형식으로 출력됩니다:
- **시뮬레이션 기간**: 시작 및 종료 날짜.
- **초기 및 최종 포트폴리오 가치**: 시뮬레이션 시작과 끝의 포트폴리오 가치.
- **포트폴리오 수익률**: 가치 변화율(퍼센트).
- **최신 주가**: 마지막 날의 TSLA 및 TSLL 종가.
- **요약 테이블**: 시작과 종료 시점의 포트폴리오 지표(비중, 현금, 주식 보유량 등).

#### 예시 출력
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

---

## Code Structure

### Key Functions
1. **`load_data(start_date, end_date)`**:
   - TSLA, TSLL, 공포탐욕지수 데이터를 로드하고 지정된 기간으로 필터링.
   - RSI, SMA(5, 10, 50, 200), MACD, Bollinger Bands, ATR, Weekly RSI, Stochastic Oscillator, OBV, VWAP, BB Width 등 기술적 지표 계산.

2. **지표 계산 함수**:
   - `calculate_rsi(series, timeperiod)`: 상대강도지수(RSI) 계산.
   - `calculate_sma(series, timeperiod)`: 단순이동평균(SMA) 계산.
   - `calculate_macd(series, fastperiod, slowperiod, signalperiod)`: MACD, 시그널 라인, 히스토그램 계산.
   - `calculate_bollinger_bands(series, timeperiod, nbdevup, nbdevdn)`: Bollinger Bands 계산.
   - `calculate_atr(df, timeperiod)`: 평균 진폭 범위(ATR) 계산.
   - `calculate_stochastic(df, k_period, d_period)`: Stochastic Oscillator %K와 %D 계산.
   - `calculate_obv(close, volume)`: 온밸런스 볼륨(OBV) 계산.
   - `calculate_vwap(df)`: 거래량 가중 평균 가격(VWAP) 계산.
   - `get_rsi_trend(rsi_series, window)`: RSI 추세(상승/하락/안정) 판단.

3. **`load_params(file_path="optimal_params.json")`**:
   - `optimal_params.json`에서 최적화된 파라미터 로드. 파일이 없거나 손상된 경우 기본값 사용.
   - **예시**:
     ```json
     {
         "version": "2.0",
         "parameters": {
             "fg_buy": 25,
             "fg_sell": 75,
             "daily_rsi_buy": 30,
             "daily_rsi_sell": 70,
             "weekly_rsi_buy": 40,
             "weekly_rsi_sell": 60,
             "volume_change_strong_buy": 0.5,
             "volume_change_weak_buy": 0.2,
             "volume_change_sell": -0.2,
             "w_strong_buy": 2.0,
             "w_weak_buy": 1.0,
             "w_sell": 1.0,
             "stochastic_buy": 20,
             "stochastic_sell": 80,
             "obv_weight": 1.0,
             "bb_width_weight": 1.0,
             "short_rsi_buy": 25,
             "short_rsi_sell": 75,
             "bb_width_low": 0.1,
             "bb_width_high": 0.2,
             "w_short_buy": 1.5,
             "w_short_sell": 1.5
         }
     }
     ```

4. **`get_target_tsll_weight(...)`**:
   - 공포탐욕지수, RSI, MACD, Bollinger Bands, 거래량 변화, Stochastic Oscillator, OBV, BB Width, 단기 지표(SMA5, SMA10, RSI5, MACD_short, VWAP)를 기반으로 TSLL 목표 비중 계산.
   - 매수/매도 신호와 가중치(`w_strong_buy`, `w_weak_buy` 등)를 활용.

5. **`adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)`**:
   - 목표 비중에 맞춰 TSLA와 TSLL을 매수/매도하며 거래 비용(0.1%) 적용.
   - 매수 시 충분한 현금이 있는 경우에만 거래 실행.

6. **`simulate_portfolio(start_date, end_date, params)`**:
   - 지정된 기간 동안 포트폴리오 시뮬레이션 수행.
   - 전일 데이터로 비중 결정(After Market)하고 당일 종가로 조정.
   - 포트폴리오 가치 추적 및 최종 결과 반환.

7. **`main()`**:
   - 명령줄 인자 파싱, 파라미터 로드, 시뮬레이션 실행, 결과 출력.

### Logic Explanation
- **포트폴리오 초기화**:
  - 초기 자산: $100,000(현금).
  - TSLA 및 TSLL 주식: 0주.

- **일일 조정**:
  1. **목표 비중 계산**: 전일 데이터를 사용하여 TSLL과 TSLA의 목표 비중 결정.
  2. **포트폴리오 재조정**:
     - 목표 주식 수 계산: `(목표 비중 × 총 포트폴리오 가치) / 당일 종가`.
     - 매수/매도 시 0.1% 거래 비용 적용.
     - 현금 잔액 조정.
  3. **포트폴리오 가치 추적**: 매일 가치 기록.

- **최종 계산**:
  - 최종 가치: `(TSLA 주식 수 × TSLA 종가) + (TSLL 주식 수 × TSLL 종가) + 현금`.
  - TSLL, TSLA, 현금 비중 계산 및 요약 테이블 출력.

- **매수/매도 조건**:
  - **매수 신호**: 낮은 공포탐욕지수, 낮은 RSI, 양의 MACD 히스토그램, 증가하는 OBV, SMA5 > SMA10 등.
  - **매도 신호**: 높은 공포탐욕지수, 높은 RSI, 음의 MACD 히스토그램, 감소하는 OBV, SMA5 < SMA10 등.

---

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 금융 전문가와 상담하세요.
