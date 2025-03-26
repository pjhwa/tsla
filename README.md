# Portfolio Pulse (`portfolio_pulse.py`)

## Overview
이 프로그램은 TSLA와 TSLL 주식으로 구성된 포트폴리오를 관리하며, 다양한 시장 지표를 분석하여 수익률을 극대화할 수 있는 최적의 비중 조정을 제안합니다. Fear & Greed Index, RSI, MACD, 볼린저 밴드 등의 지표를 활용해 동적으로 포트폴리오를 조정하며, 거래 내역이 없는 경우 초기 자산 $100,000을 기준으로 추천을 제공합니다.

## Features
- **시장 지표 분석**: Fear & Greed Index, 일일/주간 RSI, MACD, 볼린저 밴드, 거래량 변화 등을 실시간으로 분석합니다.
- **포트폴리오 상태 확인**: `transactions.txt` 파일을 통해 현재 보유 주식 수와 초기 투자액을 계산하거나, 파일이 없으면 초기 자산 $100,000을 기준으로 작동합니다.
- **동적 비중 조정**: 시장 지표를 기반으로 TSLA와 TSLL의 목표 비중을 계산하고, 매수/매도 제안을 제공합니다.
- **수익률 계산**: 초기 투자액 대비 현재 포트폴리오의 수익률을 계산하여 성과를 평가합니다.
- **로그 기록**: 조정 내역을 `portfolio_log.csv` 파일에 기록하여 추적 가능합니다.

## Installation
이 프로그램을 실행하려면 Python 3.6 이상이 필요합니다. 필요한 라이브러리를 설치하려면 다음 명령어를 실행하세요.

```bash
pip install yfinance pandas requests scipy tabulate
```

## Usage

### 1. 데이터 파일 준비
   - `transactions.txt` (선택 사항): 포트폴리오의 거래 내역을 기록한 파일입니다. 형식은 날짜 종목 액션 주식수 주가이며, 예시는 다음과 같습니다.
     ```
     2025/1/1 TSLL hold 1000 23.65
     2025/2/12 TSLL sell 1000 18.48
     2025/3/4 TSLA buy 46 265
     ```
     - 파일이 없으면 초기 자산 $100,000으로 가정합니다.
   - `optimal_params.json` (선택 사항): 최적화된 매수/매도 임계값을 포함한 JSON 파일입니다. 없으면 기본값이 사용됩니다.

### 2. 프로그램 실행
   - 저장소를 클론합니다.
     ```bash
     git clone https://github.com/pjhwa/tsla.git
     ```
   - 프로젝트 디렉토리로 이동합니다.
     ```bash
     cd tsla
     ```
   - 다음 명령어로 프로그램을 실행합니다.
     ```bash
     python3 portfolio_pulse.py
     ```
   - 프로그램은 실시간 주가 데이터와 시장 지표를 분석하여 포트폴리오 조정 제안을 출력합니다.

### 3. 출력 확인
   - 프로그램은 시장 지표, 현재 주가, 포트폴리오 상태, 추천 비중, 조정 제안, 조정 이유 등을 출력합니다.

## Requirements

### `transactions.txt` (선택 사항)
  - 거래 내역을 기록한 텍스트 파일입니다. 
  - 형식: 날짜 종목 액션 주식수 주가 (예: 2025/3/1 TSLL hold 5000 12.33)
  - hold: 초기 보유량, buy: 매수, sell: 매도.
  - 파일이 없으면 초기 자산 $100,000으로 가정합니다.
### `optimal_params.json` (선택 사항)
  - 매수/매도 임계값을 사용자 정의할 수 있는 JSON 파일입니다.
  - 기본값 예시:
    ```json
    {
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
        "w_sell": 1.0
        }

    ```

## Market Indexes and Analysis Logic
이 프로그램은 다음과 같은 시장 지표를 활용합니다.
- **Fear & Greed Index**: 시장의 공포와 탐욕 수준을 측정합니다.
- **RSI (Relative Strength Index)**: 일일 및 주간 RSI로 과매수/과매도 상태를 평가합니다.
- **MACD (Moving Average Convergence Divergence)**: 주가의 모멘텀을 분석합니다.
- **볼린저 밴드**: 주가 변동성을 측정합니다.
- **거래량 변화**: 거래량 변화를 분석하여 매수/매도 신호를 탐지합니다.
- **SMA (Simple Moving Average)**: 50일 및 200일 이동평균으로 추세를 확인합니다.

### 매수/매도 신호 조건
포트폴리오 조정은 다음과 같은 매수/매도 신호에 기반합니다.
#### 매수 신호 (Buy Signals)
- **Fear & Greed Index ≤ fg_buy (기본값: 25)**: 시장이 공포에 빠져 있을 때 매수 신호가 발생합니다.
- **Daily RSI < daily_rsi_buy (기본값: 30)**: 일일 RSI가 과매도 상태일 때 매수 신호가 발생합니다.
- **MACD > MACD Signal and MACD Signal < 0**: MACD가 신호선을 상향 돌파하고, MACD가 0 아래에 있을 때 매수 신호가 발생합니다.
- **Volume Change > volume_change_buy (기본값: 0.1)**: 거래량이 크게 증가할 때 매수 신호가 발생합니다.
- **Close < Lower Bollinger Band**: 주가가 볼린저 하단 밴드 아래로 떨어질 때 매수 신호가 발생합니다.
- **RSI Increasing and Close > SMA200**: RSI가 증가 추세이고, 주가가 200일 이동평균을 상회할 때 매수 신호가 발생합니다.
#### 매도 신호 (Sell Signals)
- **Fear & Greed Index ≥ fg_sell (기본값: 75)**: 시장이 탐욕에 빠져 있을 때 매도 신호가 발생합니다.
- **Daily RSI > daily_rsi_sell (기본값: 70)**: 일일 RSI가 과매수 상태일 때 매도 신호가 발생합니다.
- **MACD < MACD Signal and MACD Signal > 0**: MACD가 신호선을 하향 돌파하고, MACD가 0 위에 있을 때 매도 신호가 발생합니다.
- **Volume Change < volume_change_sell (기본값: -0.1)**: 거래량이 크게 감소할 때 매도 신호가 발생합니다.
- **Close > Upper Bollinger Band**: 주가가 볼린저 상단 밴드 위로 상승할 때 매도 신호가 발생합니다.
- **RSI Decreasing and Close < SMA200**: RSI가 감소 추세이고, 주가가 200일 이동평균을 하회할 때 매도 신호가 발생합니다.

### 비중 조정 로직
- 매수/매도 신호의 개수에 따라 TSLL의 비중을 동적으로 조정합니다.
- **매수 신호**가 발생할 때마다 비중을 w_buy * 0.1 (기본값: 1.5 * 0.1 = 0.15)만큼 증가시키고, **매도 신호**가 발생할 때마다 w_sell * 0.1 (기본값: 1.0 * 0.1 = 0.1)만큼 감소시킵니다.
- 예: 매수 신호 2개 발생 시 비중이 1.5 * 2 * 0.1 = 0.3만큼 증가합니다.
- 목표 비중은 0%에서 100% 사이로 제한됩니다.
- 자세한 비중 조정의 로직은 **Adjustment_Logic.md** 파일을 참고하세요.

## 출력 및 결과 해석
프로그램은 다음 정보를 출력합니다.
- **Market Indicators**: 현재 시장 지표를 테이블로 표시합니다.
- **Current Stock Prices**: TSLA와 TSLL의 최신 주가.
- **Current Portfolio**: 초기 투자액, 보유 주식 수, 주식 가치, 비중, 수익률.
- **Recommended Portfolio Weights**: 최적의 TSLA/TSLL 비중 및 필요한 주식 수.
- **Portfolio Adjustment Suggestion**: 매수/매도 제안 (차이 $100 미만은 조정 생략).
- **Adjustment Reasons**: 비중 조정 근거.

### 예시 출력
```
### Market Indicators (as of 2025-03-24)
╒══════════════════════╤═════════╤═══════════════╕
│ Indicator            │ Value   │ Trend/Notes   │
╞══════════════════════╪═════════╪═══════════════╡
│ Fear & Greed Index   │ 27.86   │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Daily RSI            │ 51.72   │ Increasing    │
├──────────────────────┼─────────┼───────────────┤
│ Weekly RSI           │ 25.12   │ -             │
├──────────────────────┼─────────┼───────────────┤
│ TSLA Close           │ $278.39 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ SMA50                │ $332.99 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ SMA200               │ $284.69 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Upper Bollinger Band │ $308.83 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Lower Bollinger Band │ $209.62 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Volume Change        │ 26.36%  │ -             │
├──────────────────────┼─────────┼───────────────┤
│ ATR                  │ $18.49  │ -             │
╘══════════════════════╧═════════╧═══════════════╛

### Current Stock Prices
- **TSLA Close**: $278.39
- **TSLL Close**: $11.16

### Current Portfolio
- Initial Investment: $90000.00
- TSLA: 0 shares, value: $0.00
- TSLL: 5000 shares, value: $60000.00
- Total Portfolio Value: $60000.00
- TSLA Weight: 0.00%
- TSLL Weight: 100.00%
- Portfolio Returns: -33.33%

### Recommended Portfolio Weights
- **TSLA Weight**: 0% (approx. 0 shares)
- **TSLL Weight**: 100% (approx. 5000 shares)

### Portfolio Adjustment Suggestion
 - No significant adjustment needed.

### Adjustment Reasons
Buy Signals:
  - MACD > MACD Signal and MACD Signal < 0
  - Volume Change > 0.24 (Strong Buy)
```

## 백테스트
과거 데이터를 기준으로 각 지표에 대한 최적의 파라미터를 찾기 위한 백테스트를 `backtest.py`로 수행할 수 있습니다. 자세한 내용은 `Backtest.md` 파일의 내용을 참고하세요.
도출된 최적의 파라미터는 `optimal_params.json` 파일로 저장되고, `portfolio_pulse.py` 실행 시 읽어들입니다.

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
