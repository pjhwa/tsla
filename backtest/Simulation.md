# Portfolio Simulation (`simulation.py`)

## Overview
`simulation.py`는 TSLA(테슬라 주식)와 TSLL(테슬라 레버리지 ETF)로 구성된 포트폴리오의 성과를 과거 특정 기간 동안 시뮬레이션하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터, 공포탐욕지수(Fear & Greed Index), 그리고 `optimal_params.json`에서 제공되는 최적화된 파라미터를 활용하여 포트폴리오의 자산 비중을 동적으로 조정합니다. 시뮬레이션 결과는 최종 포트폴리오 가치, 수익률, 주식 보유량, 현금 잔액 등을 포함하며, 이를 요약 테이블과 일일 가치 추이로 출력합니다. 또한, 각 시뮬레이션마다 상세한 로그 파일을 생성하여 매일의 활동을 기록합니다.

### Features
- **명령줄 유연성**: `--start_date`와 `--days` 인자를 통해 시뮬레이션 기간을 자유롭게 설정 가능.
- **기술적 지표**: RSI, SMA(5, 10, 50, 200), MACD, Bollinger Bands, ATR, Stochastic Oscillator, OBV, VWAP, BB Width 등 다양한 지표를 계산하여 투자 결정을 지원.
- **현금 관리**: 현금 잔액을 추적하며, 주식 매수/매도 시 0.1%의 거래 비용을 적용해 현실적인 시뮬레이션을 구현.
- **포트폴리오 재조정**: 시장 지표를 기반으로 매일 TSLA와 TSLL의 비중을 목표 비중에 맞춰 조정.
- **결과 출력**: `tabulate`를 사용해 초기 및 최종 포트폴리오 상태를 깔끔한 테이블로 출력하며, 일일 포트폴리오 가치 추이도 제공.
- **로그 기록**: 매일의 주가, 목표 비중, 조정 사유, 거래 내역, 포트폴리오 상태를 로그 파일에 상세히 기록.

---

## Requirements

### Python 버전
- **Python**: 3.6 이상

### 필요한 라이브러리
- `pandas`: 데이터 처리 및 분석
- `numpy`: 수치 계산
- `scipy`: 선형 회귀 분석(RSI 추세 계산)
- `tabulate`: 테이블 형식 출력
- `json`: 파라미터 파일 로드
- `argparse`: 명령줄 인자 파싱
- `logging`: 로그 기록

#### 설치 명령어
```bash
pip install pandas numpy scipy tabulate
```

### 데이터 파일
스크립트 실행을 위해 다음 파일이 동일 디렉토리에 있어야 합니다:
- **`fear_greed_2years.csv`**: 공포탐욕지수 데이터 (컬럼: `date`, `y`)
- **`TSLA-history-2y.csv`**: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume` 등)
- **`TSLL-history-2y.csv`**: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume` 등)
- **`optimal_params.json`**: 최적화된 파라미터 (선택 사항; 없으면 기본값 사용)

**참고**: 데이터 파일이 없는 경우 별도의 데이터 수집 스크립트를 통해 생성 가능 (예: `python3 collect_market_data.py`).

---

## Usage

### 명령줄 인자
- **`--start_date`**: 시뮬레이션 시작 날짜 (형식: `YYYY-MM-DD`). 생략 시 `--days`를 기준으로 현재 날짜에서 과거로 계산.
- **`--days`**: 시뮬레이션 기간(일 수). `--start_date`와 함께 사용 시 기간 길이를 정의하며, 단독 사용 시 현재 날짜부터 과거로 설정.

### 사용 예시
1. **특정 날짜부터 시뮬레이션**:
   ```bash
   python3 simulation.py --start_date 2024-01-01 --days 180
   ```
   - 2024년 1월 1일부터 180일 동안 시뮬레이션.

2. **과거 기간 시뮬레이션**:
   ```bash
   python3 simulation.py --days 180
   ```
   - 현재 날짜 기준 과거 180일 시뮬레이션.

### 출력 형식
시뮬레이션 결과는 터미널에 다음 형식으로 출력됩니다:
- **시뮬레이션 기간**: 시작 및 종료 날짜.
- **초기 및 최종 포트폴리오 가치**: 시뮬레이션 시작과 끝의 가치.
- **포트폴리오 수익률**: 초기 가치 대비 변화율(퍼센트).
- **최신 주가**: 마지막 날의 TSLA 및 TSLL 종가.
- **요약 테이블**: 시작과 종료 시점의 주요 지표(포트폴리오 가치, TSLL/TSLA/현금 비중, 현금액, 주식 보유량).
- **일일 포트폴리오 가치 추이**: 날짜별 포트폴리오 가치.

#### 출력 예시
```
$ python3 simulation.py --start_date 2024-01-01 --days 180

### Portfolio Simulation Results (180 days)
- Simulation Period: 2024-01-01 to 2024-06-29
- Initial Portfolio Value: $100000.00
- Final Portfolio Value: $115234.56
- Portfolio Returns: 15.23%

### Current Stock Price (2024-06-29)
- **TSLA Close**: $278.39
- **TSLL Close**: $11.16

### Summary Table
╒═════════════════╤════════════╤════════════╕
│ Metric          │ Start      │ End        │
╞═════════════════╪════════════╪════════════╡
│ Date            │ 2024-01-01 │ 2024-06-29 │
├─────────────────┼────────────┼────────────┤
│ Portfolio Value │ $100000.00 │ $115234.56 │
├─────────────────┼────────────┼────────────┤
│ TSLL Weight     │ 0.00%      │ 70.50%     │
├─────────────────┼────────────┼────────────┤
│ TSLA Weight     │ 0.00%      │ 29.40%     │
├─────────────────┼────────────┼────────────┤
│ Cash Weight     │ 100.00%    │ 0.10%      │
├─────────────────┼────────────┼────────────┤
│ Cash Amount     │ $100000.00 │ $115.23    │
├─────────────────┼────────────┼────────────┤
│ TSLA Shares     │ 0          │ 121        │
├─────────────────┼────────────┼────────────┤
│ TSLL Shares     │ 0          │ 7280       │
╘═════════════════╧════════════╧════════════╛

### Daily Portfolio Value Trend
  Date        Portfolio Value
------------  -----------------
  2024-01-01  $100000.00
  2024-01-02  $100234.56
  ...
  2024-06-29  $115234.56
```

---

## Log File
각 시뮬레이션은 `simulation-<start_date>-<end_date>.log` 형식의 로그 파일을 생성합니다 (예: `simulation-20240101-20240629.log`). 로그 파일에는 다음 내용이 포함됩니다:
- **시뮬레이션 시작 정보**: 시작 날짜, 종료 날짜, 초기 포트폴리오 가치.
- **일일 기록**:
  - 날짜
  - TSLA 및 TSLL 종가
  - 목표 TSLL/TSLA 비중
  - 조정 사유 (매수/매도 신호)
  - 실행된 거래 내역 (매수/매도 주식 수, 비용/수익)
  - 포트폴리오 가치, TSLL 비중, TSLA/TSLL 주식 수, 현금 잔액
- **시뮬레이션 종료 정보**: 최종 포트폴리오 가치 및 로그 파일 참조 안내.

### 로그 파일 예시
```
2025-03-27 15:43:39 | Simulation started: 2024-01-01 to 2024-06-29
2025-03-27 15:43:39 | Initial Portfolio Value: $100000.00
2025-03-27 15:43:39 | --- Date: 2024-01-01 ---
2025-03-27 15:43:39 | TSLA Close: $248.42, TSLL Close: $13.68
2025-03-27 15:43:39 | Target TSLL Weight: 10.00%, Target TSLA Weight: 90.00%
2025-03-27 15:43:39 | Adjustment Reasons: Fear & Greed <= fg_buy, Daily RSI < daily_rsi_buy
2025-03-27 15:43:39 | Trades Executed:
2025-03-27 15:43:39 |   - Buy 657 TSLL shares at $13.68 (Cost: $9007.18)
2025-03-27 15:43:39 |   - Buy 362 TSLA shares at $248.42 (Cost: $89968.82)
2025-03-27 15:43:39 | Portfolio Value: $100000.00, TSLL Weight: 9.00%, TSLA Shares: 362, TSLL Shares: 657, Cash: $24.00
2025-03-27 15:43:39 | --- Date: 2024-01-02 ---
2025-03-27 15:43:39 | TSLA Close: $250.00, TSLL Close: $13.80
2025-03-27 15:43:39 | Target TSLL Weight: 10.00%, Target TSLA Weight: 90.00%
2025-03-27 15:43:39 | Adjustment Reasons: None
2025-03-27 15:43:39 | No adjustment needed (Target weight unchanged)
2025-03-27 15:43:39 | Portfolio Value: $100584.60, TSLL Weight: 9.01%, TSLA Shares: 362, TSLL Shares: 657, Cash: $24.00
...
2025-03-27 15:43:40 | Simulation ended. Final Portfolio Value: $115234.56
2025-03-27 15:43:40 | See simulation-20240101-20240629.log for detailed logs.
```

---

## Code Structure

### Key Functions
1. **`load_data(start_date, end_date)`**
   - **기능**: TSLA, TSLL, 공포탐욕지수 데이터를 로드하고, 지정된 기간으로 필터링한 후 다양한 기술적 지표를 계산.
   - **지표**: RSI(14일, 5일), SMA(5, 10, 50, 200), MACD(12, 26, 9 및 5, 35, 5), Bollinger Bands, ATR, Weekly RSI, Stochastic Oscillator, OBV, VWAP, BB Width.
   - **반환**: 통합된 데이터프레임.

2. **지표 계산 함수**
   - **`calculate_rsi(series, timeperiod)`**: 상대강도지수(RSI) 계산.
   - **`calculate_sma(series, timeperiod)`**: 단순이동평균(SMA) 계산.
   - **`calculate_macd(series, fastperiod, slowperiod, signalperiod)`**: MACD, 시그널 라인, 히스토그램 계산.
   - **`calculate_bollinger_bands(series, timeperiod, nbdevup, nbdevdn)`**: Bollinger Bands 계산.
   - **`calculate_atr(df, timeperiod)`**: 평균 진폭 범위(ATR) 계산.
   - **`calculate_stochastic(df, k_period, d_period)`**: Stochastic Oscillator %K와 %D 계산.
   - **`calculate_obv(close, volume)`**: 온밸런스 볼륨(OBV) 계산.
   - **`calculate_vwap(df)`**: 거래량 가중 평균 가격(VWAP) 계산.
   - **`get_rsi_trend(rsi_series, window)`**: RSI 추세(상승/하락/안정) 판단.

3. **`load_params(file_path="optimal_params.json")`**
   - **기능**: `optimal_params.json`에서 최적화된 파라미터를 로드. 파일이 없거나 버전이 맞지 않으면 기본값 사용.
   - **기본값 예시**:
     ```json
     {
         "fg_buy": 25, "fg_sell": 75, "daily_rsi_buy": 30, "daily_rsi_sell": 70,
         "weekly_rsi_buy": 40, "weekly_rsi_sell": 60, "volume_change_strong_buy": 0.5,
         "volume_change_weak_buy": 0.2, "volume_change_sell": -0.2, "w_strong_buy": 2.0,
         "w_weak_buy": 1.0, "w_sell": 1.0, "stochastic_buy": 20, "stochastic_sell": 80,
         "obv_weight": 1.0, "bb_width_weight": 1.0, "short_rsi_buy": 25, "short_rsi_sell": 75,
         "bb_width_low": 0.1, "bb_width_high": 0.2, "w_short_buy": 1.5, "w_short_sell": 1.5
     }
     ```

4. **`get_target_tsll_weight(...)`**
   - **기능**: 공포탐욕지수, RSI, MACD, Bollinger Bands, 거래량 변화 등 다중 지표와 파라미터를 기반으로 TSLL 목표 비중을 계산.
   - **매수 조건**: 낮은 공포탐욕지수, 낮은 RSI, 양의 MACD 히스토그램, 증가하는 OBV 등.
   - **매도 조건**: 높은 공포탐욕지수, 높은 RSI, 음의 MACD 히스토그램, 감소하는 OBV 등.
   - **반환**: 목표 TSLL 비중(0~1)과 조정 사유 리스트.

5. **`adjust_portfolio(holdings, cash, target_tsla_weight, target_tsll_weight, row)`**
   - **기능**: 목표 비중에 맞춰 TSLA와 TSLL을 매수/매도하며, 0.1% 거래 비용 적용.
   - **조정 로직**:
     - TSLL 먼저 조정(매도 후 매수).
     - 남은 현금으로 TSLA 조정.
   - **반환**: 업데이트된 보유량, 현금 잔액, 거래 내역.

6. **`simulate_portfolio(start_date, end_date, params)`**
   - **기능**: 지정된 기간 동안 포트폴리오 시뮬레이션을 실행하며, 매일 목표 비중을 계산하고 조정.
   - **특징**: 비중 변화가 1% 이상일 때만 조정 실행.
   - **반환**: 초기/최종 가치, 보유량, 비중, 현금 등.

7. **`main()`**
   - **기능**: 명령줄 인자를 파싱하고, 시뮬레이션을 실행하며, 결과를 테이블로 출력.

### Logic Explanation
- **초기화**:
  - 초기 자산: $100,000(현금).
  - TSLA 및 TSLL 주식: 0주.

- **일일 조정**:
  1. **목표 비중 계산**: 당일 데이터로 TSLL/TSLA 목표 비중 결정.
  2. **포트폴리오 재조정**:
     - 목표 주식 수: `(목표 비중 × 총 포트폴리오 가치) / 당일 종가`.
     - 매수/매도 시 0.1% 거래 비용 적용.
     - 현금 잔액 업데이트.
  3. **가치 기록**: 매일 포트폴리오 가치와 상태를 로그에 기록.

- **최종 계산**:
  - 최종 가치: `(TSLA 주식 수 × TSLA 종가) + (TSLL 주식 수 × TSLL 종가) + 현금`.
  - 비중 및 결과 출력.

---

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 사용되어서는 안 됩니다. 실제 투자 결정을 내리기 전에 금융 전문가와 상담하세요.
