## Backtest 실행결과 의미 분석
Portfolio_backtest.py 실행결과는 다음과 같습니다.

```
$ ./Portfolio_backtest.py
최적 파라미터: {'daily_rsi_buy': 30, 'daily_rsi_sell': 75, 'weekly_rsi_buy': 35, 'weekly_rsi_sell': 65, 'fg_buy': 50, 'fg_sell': 70, 'volume_change_buy': 0.15, 'volume_change_sell': -0.15, 'w_buy': 1.0, 'w_sell': 2.0}
최종 포트폴리오 가치: $181341.07
최적 파라미터가 optimal_params.json에 저장되었습니다.

$ cat optimal_params.json
{"daily_rsi_buy": 30, "daily_rsi_sell": 75, "weekly_rsi_buy": 35, "weekly_rsi_sell": 65, "fg_buy": 50, "fg_sell": 70, "volume_change_buy": 0.15, "volume_change_sell": -0.15, "w_buy": 1.0, "w_sell": 2.0}
```

이 결과는 Portfolio_backtest.py 파이썬 스크립트가 과거 데이터를 이용해 백테스팅을 수행하고, 매수 및 매도 신호에 적용할 최적의 가중치를 계산한 후, 그 결과로 포트폴리오의 성과를 도출한 것입니다. 수행 결과로 도출된 최적의 파라미터는 optimal_params.json 파일에 저장됩니다. 각 항목의 의미를 아래에서 자세히 설명합니다.

### 최적 매수 가중치 (w_buy): 1.5
의미: 매수 신호가 발생했을 때 포트폴리오에서 TSLL(Tesla 관련 레버리지 ETF)의 비중을 기본 조정량(예: 0.1%)에 1.5배를 곱한 만큼 증가시킵니다. 즉, 매수 신호가 강력할 때 더 공격적으로 투자하는 전략을 나타냅니다.
예시: 기본 비중 조정량이 0.1%라면, 매수 신호 발생 시 TSLL 비중을 0.15% 증가시킵니다.

### 최적 매도 가중치 (w_sell): 1.0
의미: 매도 신호가 발생했을 때 포트폴리오에서 TSLL의 비중을 기본 조정량(예: 0.1%)에 1.0배를 곱한 만큼 감소시킵니다. 이는 매도 신호에 대해 보수적인 접근을 취한다는 뜻입니다.
예시: 기본 비중 조정량이 0.1%라면, 매도 신호 발생 시 TSLL 비중을 0.1% 감소시킵니다.

### 최종 포트폴리오 가치: $347,418.96
의미: 초기 자산 $100,000에서 시작하여 백테스팅 기간 동안 위의 전략을 적용한 후의 포트폴리오 가치입니다. 약 $347,418.96까지 증가했다는 뜻입니다.

### 총 수익률: 247.42%
의미: 초기 자산 $100,000 대비 최종 포트폴리오 가치가 247.42% 증가했다는 것을 나타냅니다. 즉, 초기 자산이 약 3.47배로 성장한 결과입니다.

## 결과 요약
이 백테스팅 결과는 과거 데이터를 기반으로 매수 신호에 더 큰 가중치(1.5)를 부여하고, 매도 신호에는 기본 가중치(1.0)를 부여했을 때 수익률이 247.42%로 극대화되었다는 것을 보여줍니다.
따라서 이 가중치(w_buy = 1.5, w_sell = 1.0)는 수익 극대화를 위한 최적화된 값으로 간주됩니다.

### 각 지표와 가중치의 관계
사용자가 언급한 지표들(Daily RSI, Weekly RSI, TSLA Close, SMA50, SMA200, Upper Bollinger Band, Lower Bollinger Band, Volume Change, Fear & Greed Index)은 포트폴리오 조정에서 매수/매도 신호를 판단하는 데 사용됩니다. 그러나 백테스팅 결과에서 제공된 가중치(w_buy와 w_sell)는 개별 지표에 직접 적용되는 것이 아니라, 매수/매도 신호의 강도에 따라 포트폴리오 비중을 조정하는 데 사용됩니다.

#### 지표의 역할
각 지표는 매수 또는 매도 신호를 생성하는 데 기여합니다. 예를 들어:
Daily RSI < 30: 과매도 상태로 매수 신호.
Fear & Greed Index ≥ 75: 극단적 탐욕 상태로 매도 신호.
TSLA Close < Lower Bollinger Band: 매수 신호.

#### 가중치의 역할
매수 신호가 발생하면 w_buy = 1.5를 적용해 비중을 더 크게 늘리고, 매도 신호가 발생하면 w_sell = 1.0을 적용해 비중을 기본 수준으로 줄입니다.
즉, 개별 지표 자체에 가중치가 부여되는 것이 아니라, 지표들이 생성한 신호에 기반하여 포트폴리오 비중을 조정할 때 가중치가 적용됩니다.

## Backtest를 위해 필요한 데이터
TSLA 과거 1년간의 데이터: TSLA-historicaldata.csv
TSLL 과거 1년간의 데이터: TSLL-historicaldata.csv
Fear & Greed Index 과거 1년간의 데이터: fear_greed_1year.csv
