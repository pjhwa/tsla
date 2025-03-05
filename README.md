아래는 사용자가 요청한 주식 데이터 분석 및 포트폴리오 조정 프로그램을 GitHub에 공개하기 위한 README.md 파일입니다. 이 파일은 프로그램의 목적, 주요 기능, 설치 방법, 사용 방법, 로직, 사용 예시, 그리고 면책 조항을 포함하고 있습니다.

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
git clone https://github.com/yourusername/your-repo-name.git
```

프로젝트 디렉토리로 이동합니다.

```bash
cd your-repo-name
```

프로그램을 실행합니다.

```bash
python portfolio_adjustment.py
```

프로그램은 시장 지표, 추천 포트폴리오 비중, 조정 이유를 출력합니다.

## Logic
포트폴리오 비중 조정은 아래 로직을 따릅니다:

### Fear & Greed Index:
≤ 25: TSLL 비중 10% 증가 (매수 신호)
≥ 75: TSLL 비중 10% 감소 (매도 신호)
### RSI:
< 30: TSLL 비중 10% 증가 (매수 신호)
70: TSLL 비중 10% 감소 (매도 신호)
### Bollinger Bands:
주가가 하단 밴드 미만: TSLL 비중 10% 증가 (매수 신호)
주가가 상단 밴드 초과: TSLL 비중 10% 감소 (매도 신호)
### Volume Change:
거래량이 10% 이상 증가: TSLL 비중 10% 증가 (매수 신호)
거래량이 10% 이상 감소: TSLL 비중 10% 감소 (매도 신호)
### MACD:
MACD가 신호선을 상향 돌파하고 신호선이 0 미만: TSLL 비중 10% 증가 (매수 신호)
MACD가 신호선을 하향 돌파하고 신호선이 0 초과: TSLL 비중 10% 감소 (매도 신호)

최종 비중은 0%에서 100% 사이로 제한됩니다.

## Example Output
프로그램 실행 시 다음과 같은 출력이 표시됩니다.

```bash
### Market Indicators (as of 2023-10-05)
╒═════════════════════════╤═══════╤═════════════╕
│ Indicator               │ Value │ Trend/Notes │
╞═════════════════════════╪═══════╪═════════════╡
│ Fear & Greed Index      │ 24.37 │ -           │
├─────────────────────────┼───────┼─────────────┤
│ Daily RSI               │ 28.81 │ Decreasing  │
├─────────────────────────┼───────┼─────────────┤
│ TSLA Close              │ $284.65 │ -           │
├─────────────────────────┼───────┼─────────────┤
│ SMA50                   │ $385.71 │ -           │
├─────────────────────────┼───────┼─────────────┤
│ SMA200                  │ $279.33 │ -           │
├─────────────────────────┼───────┼─────────────┤
│ Upper Bollinger Band    │ $408.26 │ -           │
├─────────────────────────┼───────┼─────────────┤
│ Lower Bollinger Band    │ $272.55 │ -           │
├─────────────────────────┼───────┼─────────────┤
│ Volume Change           │ -0.48% │ -           │
╘═════════════════════════╧═══════╧═════════════╛

### Recommended Portfolio Weights
- **TSLA Weight**: 80%
- **TSLL Weight**: 20%

### Portfolio Adjustment
- Buy TSLL: $20000.00

### Adjustment Reasons
- Buy Signal: RSI < 30 or MACD > MACD Signal
- Fear & Greed Index ≤ 25: Extreme Fear
```

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
