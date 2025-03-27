# Portfolio Pulse (`portfolio_pulse.py`)

## Overview
**Portfolio Pulse**는 TSLA와 TSLL 주식으로 구성된 포트폴리오를 관리하고 최적화하는 Python 프로그램입니다. 실시간 시장 데이터를 기반으로 다양한 기술적 지표를 분석하여 포트폴리오의 수익률을 극대화할 수 있는 동적 비중 조정 제안을 제공합니다. Fear & Greed Index, RSI, MACD, Bollinger Bands, Stochastic Oscillator, OBV, ATR, 거래량 변화 등 기존 지표 외에도 단기 예측을 위한 SMA5, SMA10, RSI5, 단기 MACD, VWAP 등의 지표를 추가하여 After Market 반응성을 강화했습니다. `transactions.txt` 파일을 통해 실제 거래 내역을 반영하거나, 파일이 없으면 초기 자산 $100,000을 기준으로 작동합니다. 또한, 거래 비용(0.1%)을 반영하여 현실적인 수익률을 평가합니다.

---

## Features
- **실시간 시장 지표 분석**: Fear & Greed Index, 일일/주간 RSI, MACD, Bollinger Bands, Stochastic Oscillator, OBV, ATR, 거래량 변화, 단기 SMA(5일, 10일), 단기 RSI(5일), 단기 MACD, VWAP 등 다양한 지표를 분석합니다.
- **포트폴리오 상태 확인**: `transactions.txt` 파일을 통해 보유 주식 수와 초기 투자액을 계산하거나, 파일이 없으면 초기 자산 $100,000으로 가정합니다.
- **동적 비중 조정 제안**: 시장 지표를 기반으로 TSLA와 TSLL의 최적 비중을 계산하고, 구체적인 매수/매도 제안을 제공합니다.
- **수익률 및 손익 계산**: 초기 투자액 대비 현재 포트폴리오 가치, 손익, 수익률을 상세히 계산하며, 거래 비용을 반영합니다.
- **로그 기록**: 포트폴리오 조정 내역을 `portfolio_log.csv`에 기록하여 추적 가능합니다.
- **After Market 반응성 강화**: 단기 지표(SMA5, SMA10, RSI5, 단기 MACD, VWAP)를 추가하여 단기 시장 변동에 민감하게 대응합니다.
- **거래 비용 반영**: 매수/매도 시 0.1%의 거래 비용을 적용하여 현실적인 수익률을 평가합니다.

---

## Installation
이 프로그램을 실행하려면 Python 3.6 이상이 필요합니다. 필요한 라이브러리를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install yfinance pandas requests scipy tabulate numpy
```

---

## Usage

### 1. 데이터 파일 준비
- **`transactions.txt`** (선택 사항): 포트폴리오의 거래 내역을 기록한 텍스트 파일입니다. 형식은 `날짜 종목 액션 주식수 주가`이며, 예시는 다음과 같습니다:
  ```
  2025/1/1 TSLL hold 1000 23.65
  2025/2/12 TSLL sell 1000 18.48
  2025/3/4 TSLA buy 46 265
  ```
  - 파일이 없으면 초기 자산 $100,000으로 가정합니다.
- **`optimal_params.json`** (선택 사항): 최적화된 매수/매도 임계값을 포함한 JSON 파일입니다. 없으면 VIX에 따라 동적으로 설정된 기본값이 사용됩니다.

### 2. 프로그램 실행
- 저장소를 클론합니다:
  ```bash
  git clone https://github.com/pjhwa/tsla.git
  ```
- 프로젝트 디렉토리로 이동합니다:
  ```bash
  cd tsla
  ```
- 프로그램을 실행합니다:
  ```bash
  python3 portfolio_pulse.py
  ```
- 프로그램은 실시간 주가 데이터와 시장 지표를 분석하여 포트폴리오 조정 제안을 출력합니다.

### 3. 출력 확인
프로그램은 다음 정보를 출력합니다:
- **Market Indicators**: 현재 시장 지표와 트렌드/노트를 테이블로 표시.
- **Current Stock Prices**: TSLA와 TSLL의 최신 주가 및 전일 대비 변화율.
- **Current Portfolio**: 초기 투자액, 현재 포트폴리오 가치, 손익, 보유 주식 수, 비중, 수익률.
- **Previous Recommendation**: 이전 추천 비중(있을 경우).
- **Difference from Previous Recommendation**: 현재 비중과 이전 추천의 차이.
- **Recommended Portfolio Weights**: 최적의 TSLA/TSLL 비중.
- **Portfolio Adjustment Suggestion**: 매수/매도 제안.
- **Adjustment Reasons**: 비중 조정 근거.

---

## Requirements

### **`transactions.txt`** (선택 사항)
- 거래 내역을 기록한 텍스트 파일입니다.
- 형식: `날짜 종목 액션 주식수 주가` (예: `2025/3/1 TSLL hold 5000 12.33`)
- 액션: `hold` (초기 보유량), `buy` (매수), `sell` (매도).
- 파일이 없으면 초기 자산 $100,000으로 가정합니다.

### **`optimal_params.json`** (선택 사항)
- 매수/매도 임계값을 사용자 정의할 수 있는 JSON 파일입니다.
- 형식 예시:
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
- 파일이 없거나 버전이 맞지 않으면 VIX에 따라 동적 기본값이 적용됩니다.

---

## Market Indicators and Analysis Logic
이 프로그램은 다음 시장 지표를 활용하여 포트폴리오 조정을 제안합니다:

- **Fear & Greed Index**: 시장의 공포와 탐욕 수준을 측정.
- **RSI (Relative Strength Index)**: 일일 및 주간 RSI, 단기 RSI(5일)로 과매수/과매도 상태를 평가.
- **MACD (Moving Average Convergence Divergence)**: 모멘텀 분석(MACD Histogram 포함), 단기 MACD 추가.
- **Bollinger Bands**: 주가 변동성 측정(BB Width 포함).
- **Stochastic Oscillator**: 단기 과매수/과매도 상태 분석.
- **OBV (On-Balance Volume)**: 거래량 기반 추세 확인.
- **ATR (Average True Range)**: 시장 변동성 측정 및 동적 조정.
- **SMA (Simple Moving Average)**: 5일, 10일, 50일, 200일 이동평균으로 단기 및 장기 추세 확인.
- **Volume Change**: 거래량 변화 분석.
- **VWAP (Volume Weighted Average Price)**: 거래량 가중 평균 가격으로 매수/매도 압력 평가.

### 매수/매도 신호 조건
포트폴리오 조정은 다음 매수/매도 신호에 기반합니다:

#### 매수 신호 (Buy Signals)
- Fear & Greed Index ≤ `fg_buy`
- Daily RSI < `daily_rsi_buy`
- Weekly RSI < `weekly_rsi_buy`
- MACD > MACD Signal and MACD Signal < 0
- MACD Histogram > 0
- Volume Change > `volume_change_strong_buy` (Strong Buy)
- Volume Change > `volume_change_weak_buy` (Weak Buy)
- Close < Lower Bollinger Band
- RSI Increasing and Close > SMA200
- Stochastic %K < `stochastic_buy`
- OBV Increasing
- BB Width < `bb_width_low`
- SMA5 > SMA10
- Short RSI < `short_rsi_buy`
- Short MACD > Signal
- Close > VWAP

#### 매도 신호 (Sell Signals)
- Fear & Greed Index ≥ `fg_sell`
- Daily RSI > `daily_rsi_sell`
- Weekly RSI > `weekly_rsi_sell`
- MACD < MACD Signal and MACD Signal > 0
- MACD Histogram < 0
- Volume Change < `volume_change_sell`
- Close > Upper Bollinger Band
- RSI Decreasing and Close < SMA200
- Stochastic %K > `stochastic_sell`
- OBV Decreasing
- BB Width > `bb_width_high`
- SMA5 < SMA10
- Short RSI > `short_rsi_sell`
- Short MACD < Signal
- Close < VWAP

### 비중 조정 로직
- **ATR 기반 동적 조정**: 거래량 변화 임계값은 ATR로 정규화되어 변동성에 따라 조정됩니다.
- **가중치 적용**: 매수/매도 신호의 개수와 가중치(`w_strong_buy`, `w_weak_buy`, `w_sell`, `obv_weight`, `bb_width_weight`, `w_short_buy`, `w_short_sell`)를 기반으로 TSLL 비중을 계산합니다.
- **조정 공식**:
  - 증가: `(w_strong_buy * 강한 매수 신호 수 + w_weak_buy * 약한 매수 신호 수 + w_weak_buy * 기타 매수 신호 수 + obv_weight * OBV 증가 + bb_width_weight * BB Width 낮음 + w_short_buy * 단기 매수 신호 수) * 0.1`
  - 감소: `(w_sell * 매도 신호 수 + obv_weight * OBV 감소 + bb_width_weight * BB Width 높음 + w_short_sell * 단기 매도 신호 수) * 0.1`
- 목표 비중은 0%에서 100% 사이로 제한됩니다.
- 자세한 내용은 `Adjustment_Logic.md` 파일을 참고하세요.

---

## Output and Interpretation
프로그램은 다음 정보를 출력합니다:

### 예시 출력
```
### Market Indicators (as of 2025-03-25)
╒══════════════════════╤═════════════╤═══════════════════════════╕
│ Indicator            │ Value       │ Trend/Notes               │
╞══════════════════════╪═════════════╪═══════════════════════════╡
│ Fear & Greed Index   │ 28.77       │ Low                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Daily RSI            │ 52.42       │ Increasing                │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Weekly RSI           │ 27.59       │ Oversold                  │
├──────────────────────┼─────────────┼───────────────────────────┤
│ TSLA Close           │ $288.14     │ Below SMA50, Above SMA200 │
├──────────────────────┼─────────────┼───────────────────────────┤
│ SMA5                 │ $285.00     │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ SMA10                │ $290.00     │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ SMA50                │ $330.86     │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ SMA200               │ $285.25     │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Upper Bollinger Band │ $305.76     │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Lower Bollinger Band │ $211.22     │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Volume Change        │ -11.79%     │ Increasing                │
├──────────────────────┼─────────────┼───────────────────────────┤
│ ATR                  │ $18.85      │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Stochastic %K        │ 99.92       │ Above %D                  │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Stochastic %D        │ 81.71       │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ OBV                  │ 21546747900 │ Increasing                │
├──────────────────────┼─────────────┼───────────────────────────┤
│ BB Width             │ 0.3657      │ High                      │
├──────────────────────┼─────────────┼───────────────────────────┤
│ MACD Histogram       │ 8.13        │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Short RSI (5-day)    │ 60.00       │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ Short MACD           │ 2.50        │ N/A                       │
├──────────────────────┼─────────────┼───────────────────────────┤
│ VWAP                 │ $290.00     │ N/A                       │
╘══════════════════════╧═════════════╧═══════════════════════════╛

### Current Stock Prices
- **TSLA Close**: $288.14 (Change: 3.50%)
- **TSLL Close**: $11.85 (Change: 6.95%)

### Current Portfolio
- Initial Investment: $90000.00
- Current Total Value: $60000.00
- Profit/Loss: $-30000.00 (-33.33%)
- TSLA: 0 shares, value: $0.00 (0.00%)
- TSLL: 5000 shares, value: $60000.00 (100.00%)

### Previous Recommendation (as of 2025-03-24)
- TSLA: 10.00%
- TSLL: 90.00%

### Difference from Previous Recommendation
- TSLA Weight Difference: 10.00%
- TSLL Weight Difference: 10.00%

### Recommended Portfolio Weights
- TSLA: 0.00%
- TSLL: 100.00%

### Portfolio Adjustment Suggestion
- TSLA: No adjustment needed
- TSLL: No adjustment needed

### Adjustment Reasons
Buy Signals (Potential increase in TSLL weight):
  - Weekly RSI < 40
  - MACD > Signal (Signal < 0)
  - MACD Histogram > 0
  - RSI Increasing & Close > SMA200
  - OBV Increasing
  - SMA5 > SMA10
  - Short MACD > Signal
  - Close > VWAP
Sell Signals (Potential decrease in TSLL weight):
  - Stochastic %K > 80
  - BB Width > 0.2
```

---

## Backtesting and Simulation
- **백테스트 (`backtest.py`)**: 과거 데이터를 기반으로 최적의 파라미터를 도출하여 `optimal_params.json`에 저장합니다.
- **시뮬레이션 (`simulation.py`)**: 특정 기간 동안의 수익률을 시뮬레이션하여 전략의 성과를 평가합니다.

---

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하세요.
