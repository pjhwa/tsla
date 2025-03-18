# TSLA & TSLL Portfolio Adjustment Program

## Overview
이 프로그램은 TSLA와 TSLL의 주식 데이터를 분석하여 다양한 기술적 지표를 기반으로 포트폴리오 비중을 조정합니다. 사용되는 지표에는 Fear & Greed Index, RSI, Bollinger Bands, 거래량 변화 등이 포함되며, 이를 통해 매수/매도 신호를 제공하고 TSLA와 TSLL의 최적 비중을 추천합니다.

## Features
- CNN에서 실시간 Fear & Greed Index를 수집
- `yfinance`를 통해 TSLA와 TSLL의 주식 데이터 수집
- RSI, SMA, Bollinger Bands, MACD 등 기술적 지표 계산
- 미리 정의된 조건에 따라 포트폴리오 비중 조정
- 시장 지표, 추천 비중, 조정 이유 출력

## Installation
프로그램을 실행하려면 아래 Python 라이브러리를 설치해야 합니다.

```bash
pip install yfinance pandas requests scipy tabulate
```

## Usage
저장소를 클론합니다.

```bash
git clone https://github.com/pjhwa/tsla.git
```

프로젝트 디렉토리로 이동합니다.

```bash
cd tsla
```

프로그램을 실행합니다.

```bash
python TSLA_Portfolio.py
```

프로그램은 시장 지표, 추천 포트폴리오 비중, 조정 이유를 출력합니다.

## 지표 계산

### RSI
14일 기준 상대강도지수를 계산합니다.
### SMA
50일 및 200일 단순 이동 평균을 계산합니다.
### Bollinger Bands
20일 기준 상단/하단 밴드를 계산합니다.
### MACD
12일, 26일, 9일 기준으로 계산합니다.
### RSI Trend
최근 10일 데이터를 통해 상승/하락/안정 추세를 판단합니다.

## 매수/매도 신호 판단

### 매수 조건
Fear & Greed Index ≤ 25 (극단적 공포)
Daily RSI < 30 (과매도)
MACD가 시그널선을 상향 돌파
거래량 10% 증가
주가가 하단 볼린저 밴드 이하
RSI 상승 추세 + 주가 > SMA200

### 매도 조건
Fear & Greed Index ≥ 75 (극단적 탐욕)
Daily RSI > 70 (과매수)
MACD가 시그널선을 하향 돌파
거래량 10% 감소
주가가 상단 볼린저 밴드 이상
RSI 하락 추세 + 주가 < SMA200

### 포트폴리오 조정
매수 신호 개수에 w_buy = 1.5를 곱해 비중을 증가시키고, 매도 신호 개수에 w_sell = 1.0을 곱해 비중을 감소시킵니다.
비중은 0%에서 100% 사이로 제한됩니다.

## Example Output
프로그램 실행 시 다음과 같은 출력이 표시됩니다.

```bash
$ ./TSLA_Portfolio.py
데이터 로드 중...

### Market Indicators (as of 2025-03-17)
╒══════════════════════╤═════════╤═══════════════╕
│ Indicator            │ Value   │ Trend/Notes   │
╞══════════════════════╪═════════╪═══════════════╡
│ Fear & Greed Index   │ 22.06   │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Daily RSI            │ 31.11   │ Increasing    │
├──────────────────────┼─────────┼───────────────┤
│ Weekly RSI           │ 14.97   │ -             │
├──────────────────────┼─────────┼───────────────┤
│ TSLA Close           │ $238.01 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ SMA50                │ $348.30 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ SMA200               │ $282.99 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Upper Bollinger Band │ $371.22 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Lower Bollinger Band │ $198.51 │ -             │
├──────────────────────┼─────────┼───────────────┤
│ Volume Change        │ 11.11%  │ -             │
╘══════════════════════╧═════════╧═══════════════╛

### Current Stock Prices
- **TSLA Close**: $238.01
- **TSLL Close**: $8.34

### Recommended Portfolio Weights
- **TSLA Weight**: 70%
- **TSLL Weight**: 30%

### Portfolio Adjustment
 - Buy TSLL: $30000.00

### Adjustment Reasons
- Buy Signal: Fear & Greed Index ≤ 25, Volume Change > 10%
```

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
