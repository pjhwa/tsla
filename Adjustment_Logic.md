# Portfolio Weights Adjustment Logic (`portfolio_pulse.py`)

## **비중 조정 로직의 개요**

`portfolio_pulse.py`는 TSLA와 TSLL로 구성된 포트폴리오의 비중을 시장 상황에 따라 동적으로 조정하는 프로그램입니다. 이 프로그램은 다양한 시장 지표를 분석하여 TSLL과 TSLA의 목표 비중을 계산하고, 현재 보유 주식 수와 비교하여 구체적인 매수/매도 제안을 제공함으로써 포트폴리오의 수익률을 극대화하는 것을 목표로 합니다.

비중 조정 로직은 다음과 같은 단계로 진행됩니다:

1. **시장 지표 수집 및 분석**
2. **최적 파라미터 로드**
3. **현재 포트폴리오 상태 파악**
4. **목표 TSLL 비중 계산**
5. **목표 TSLA 비중 계산**
6. **필요한 주식 수 계산**
7. **포트폴리오 조정 제안**
8. **조정 이유 제공**

이제 각 단계를 코드와 함께 상세히 설명하겠습니다.

---

## **1. 시장 지표 수집 및 분석**

시장 상황을 평가하기 위해 다양한 기술적 지표와 심리 지표를 수집합니다. 이 지표들은 TSLL과 TSLA의 비중 조정에 핵심적인 입력값으로 사용됩니다.

### **수집되는 지표**
- **Fear & Greed Index**: CNN에서 제공하는 시장 심리 지표로, 공포(0)에서 탐욕(100)까지의 값을 가집니다. `get_fear_greed_index` 함수로 최신 값을 가져옵니다.
- **RSI (Relative Strength Index)**:
  - 일일 RSI: `get_stock_data`에서 TSLA의 14일 RSI를 계산합니다.
  - 주간 RSI: `get_weekly_rsi`에서 TSLA의 주간 데이터를 기반으로 계산합니다.
  - 단기 RSI: 5일 RSI를 추가로 계산하여 단기 과매수/과매도 상태를 분석합니다.
- **SMA (Simple Moving Average)**: TSLA의 5일, 10일, 50일, 200일 단순 이동평균을 `calculate_sma`로 계산하여 단기 및 장기 추세를 분석합니다.
- **Bollinger Bands**: `calculate_bollinger_bands`로 20일 SMA와 상단/하단 밴드(표준편차 2배)를 계산하여 가격 변동성을 측정합니다.
- **MACD (Moving Average Convergence Divergence)**:
  - 표준 MACD: `calculate_macd`로 12일/26일 EMA 차이와 9일 신호선을 계산합니다.
  - 단기 MACD: 5일/35일/5일 설정으로 단기 추세 전환을 포착합니다.
- **Volume Change**: 거래량 변화율(`pct_change`)을 계산하여 시장 참여자의 관심도를 파악합니다.
- **ATR (Average True Range)**: `calculate_atr`로 14일 평균 진폭을 계산하여 가격 변동성을 반영합니다.
- **Stochastic Oscillator**: `calculate_stochastic`로 %K와 %D를 계산하여 단기 과매수/과매도 상태를 분석합니다.
- **OBV (On-Balance Volume)**: `calculate_obv`로 거래량을 기반으로 추세의 강도를 측정합니다.
- **BB Width**: Bollinger Bands의 너비를 계산하여 변동성의 크기를 평가합니다.
- **VWAP (Volume Weighted Average Price)**: `calculate_vwap`로 거래량 가중 평균 가격을 계산하여 매수/매도 압력을 평가합니다.

### **코드 예시**
```python
def get_stock_data(ticker, period="max", interval="1d"):
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['RSI5'] = calculate_rsi(df['Close'], 5)
    df['SMA5'] = calculate_sma(df['Close'], 5)
    df['SMA10'] = calculate_sma(df['Close'], 10)
    df['SMA50'] = calculate_sma(df['Close'], 50)
    df['SMA200'] = calculate_sma(df['Close'], 200)
    df['Upper Band'], df['Middle Band'], df['Lower Band'] = calculate_bollinger_bands(df['Close'])
    df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'], 12, 26, 9)
    df['MACD_short'], df['MACD_signal_short'] = calculate_macd(df['Close'], 5, 35, 5)
    df['Volume Change'] = df['Volume'].pct_change()
    df['ATR'] = calculate_atr(df, 14)
    df['Stochastic_K'], df['Stochastic_D'] = calculate_stochastic(df)
    df['OBV'] = calculate_obv(df)
    df['BB_width'] = (df['Upper Band'] - df['Lower Band']) / df['Middle Band']
    df['VWAP'] = calculate_vwap(df)
    return df
```
- TSLA와 TSLL의 데이터를 `yfinance`로 가져오며, 결측치는 선형 보간(`interpolate`)으로 처리됩니다.

---

## **2. 최적 파라미터 로드**

비중 조정에 필요한 임계값과 가중치는 `optimal_params.json` 파일에서 로드됩니다. 이 파일은 백테스트를 통해 도출된 최적 파라미터를 포함하며, 파일이 없거나 버전이 맞지 않으면 VIX에 따라 동적으로 설정된 기본값이 사용됩니다.

### **파라미터 예시 (`optimal_params.json`)**
```json
{
    "version": "2.0",
    "parameters": {
        "fg_buy": 21.57,
        "fg_sell": 64.75,
        "daily_rsi_buy": 38.97,
        "daily_rsi_sell": 79.21,
        "weekly_rsi_buy": 34.78,
        "weekly_rsi_sell": 75.94,
        "volume_change_strong_buy": 0.31,
        "volume_change_weak_buy": 0.34,
        "volume_change_sell": -0.5,
        "w_strong_buy": 1.78,
        "w_weak_buy": 1.99,
        "w_sell": 1.78,
        "stochastic_buy": 20.20,
        "stochastic_sell": 74.37,
        "obv_weight": 0.70,
        "bb_width_weight": 1.38,
        "short_rsi_buy": 26.95,
        "short_rsi_sell": 70.26,
        "bb_width_low": 0.12,
        "bb_width_high": 0.30,
        "w_short_buy": 0.95,
        "w_short_sell": 1.64
    }
}
```
- **주요 파라미터**:
  - `fg_buy`, `fg_sell`: Fear & Greed Index의 매수/매도 임계값.
  - `daily_rsi_buy`, `daily_rsi_sell`: 일일 RSI의 매수/매도 임계값.
  - `weekly_rsi_buy`, `weekly_rsi_sell`: 주간 RSI의 매수/매도 임계값.
  - `volume_change_strong_buy`, `volume_change_weak_buy`, `volume_change_sell`: 거래량 변화율 임계값.
  - `w_strong_buy`, `w_weak_buy`, `w_sell`: 신호 강도에 따른 가중치.
  - `stochastic_buy`, `stochastic_sell`: Stochastic Oscillator의 매수/매도 임계값.
  - `obv_weight`, `bb_width_weight`: OBV와 BB Width 신호의 가중치.
  - `short_rsi_buy`, `short_rsi_sell`: 단기 RSI(5일)의 매수/매도 임계값.
  - `bb_width_low`, `bb_width_high`: BB Width의 낮음/높음 기준.
  - `w_short_buy`, `w_short_sell`: 단기 매수/매도 가중치.

### **코드 동작**
```python
def load_optimal_params(file_path="optimal_params.json", latest_version="2.0"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if "version" in data and "parameters" in data and data["version"] == latest_version:
                return data["parameters"]
            print("Parameters are outdated or format is incorrect. Using default values.")
    except Exception:
        print("Failed to load parameter file. Using dynamic default values.")
    return get_dynamic_default_params(get_current_vix())
```
- VIX 값에 따라 기본값을 동적으로 설정하며, 예를 들어 VIX > 30이면 더 보수적인 임계값을 사용합니다.

---

## **3. 현재 포트폴리오 상태 파악**

현재 TSLA와 TSLL의 보유 주식 수와 비중을 확인합니다.

### **코드 동작**
- **`load_transactions`**: `transactions.txt`에서 거래 내역을 읽어 보유 주식 수를 계산합니다.
  ```python
  def load_transactions(file_path="transactions.txt"):
      if not os.path.exists(file_path):
          print("transactions.txt file not found. Using initial assets.")
          return {}, 0
      df = pd.read_csv(file_path, sep='\s+', names=["date", "ticker", "action", "shares", "stock_price"])
      holdings = {}
      initial_investment = 0
      for i, row in df.iterrows():
          if i == 0 and row['action'] == "hold":
              initial_investment = row['shares'] * row['stock_price']
              holdings[row['ticker']] = row['shares']
          elif row['action'] == "buy":
              holdings[row['ticker']] = holdings.get(row['ticker'], 0) + row['shares']
          elif row['action'] == "sell" and row['ticker'] in holdings:
              holdings[row['ticker']] -= row['shares']
              if holdings[row['ticker']] <= 0:
                  del holdings[row['ticker']]
      return {k: v for k, v in holdings.items() if v > 0}, initial_investment
  ```
- **`calculate_portfolio_metrics`**: 현재 주가를 기반으로 포트폴리오 가치와 비중을 계산합니다.
  ```python
  def calculate_portfolio_metrics(current_holdings, tsla_close, tsll_close, initial_investment):
      tsla_shares = current_holdings.get("TSLA", 0)
      tsll_shares = current_holdings.get("TSLL", 0)
      tsla_value = tsla_shares * tsla_close
      tsll_value = tsll_shares * tsll_close
      total_value = tsla_value + tsll_value
      tsla_weight = tsla_value / total_value if total_value > 0 else 0
      tsll_weight = tsll_value / total_value if total_value > 0 else 0
      returns = ((total_value - initial_investment) / initial_investment * 100) if initial_investment > 0 else 0
      return total_value, tsla_value, tsll_value, tsla_weight, tsll_weight, returns
  ```

---

## **4. 목표 TSLL 비중 계산**

`get_target_tsll_weight` 함수는 시장 지표와 최적 파라미터를 활용하여 TSLL의 목표 비중을 계산합니다. 이 과정은 비중 조정의 핵심입니다.

### **세부 단계**

#### **4.1. 동적 임계값 계산**
- ATR을 사용하여 변동성에 따라 거래량 변화율 임계값을 조정합니다.
  ```python
  atr_normalized = atr / close if close > 0 else 0
  volume_change_strong_buy = optimal_params["volume_change_strong_buy"] * (1 + atr_normalized)
  volume_change_weak_buy = optimal_params["volume_change_weak_buy"] * (1 + atr_normalized)
  volume_change_sell = optimal_params["volume_change_sell"] * (1 + atr_normalized)
  ```
- 변동성이 높을수록 더 큰 거래량 변화를 요구합니다.

#### **4.2. 매수 및 매도 조건 평가**
- **매수 조건**:
  - Fear & Greed Index ≤ `fg_buy`
  - Daily RSI < `daily_rsi_buy`
  - Weekly RSI < `weekly_rsi_buy`
  - MACD > Signal and Signal < 0
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

- **매도 조건**:
  - Fear & Greed Index ≥ `fg_sell`
  - Daily RSI > `daily_rsi_sell`
  - Weekly RSI > `weekly_rsi_sell`
  - MACD < Signal and Signal > 0
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

  ```python
  buy_conditions = {
      f"Fear & Greed Index ≤ {optimal_params['fg_buy']}": fear_greed <= optimal_params["fg_buy"],
      f"Daily RSI < {optimal_params['daily_rsi_buy']}": daily_rsi < optimal_params["daily_rsi_buy"],
      # ... (다른 조건들)
  }
  sell_conditions = {
      f"Fear & Greed Index ≥ {optimal_params['fg_sell']}": fear_greed >= optimal_params["fg_sell"],
      # ... (다른 조건들)
  }
  buy_reasons = [cond for cond, val in buy_conditions.items() if val]
  sell_reasons = [cond for cond, val in sell_conditions.items() if val]
  ```

#### **4.3. 가중치 적용 및 비중 조정**
- **가중치 계산**:
  - Strong Buy: `w_strong_buy`
  - Weak Buy 및 기타 매수 신호: `w_weak_buy`
  - 매도 신호: `w_sell`
  - OBV 및 BB Width: `obv_weight`, `bb_width_weight`
  - 단기 매수/매도: `w_short_buy`, `w_short_sell`

  ```python
  strong_buy_count = sum(1 for r in buy_reasons if "Strong Buy" in r)
  weak_buy_count = sum(1 for r in buy_reasons if "Weak Buy" in r and "Strong Buy" not in r)
  other_buy_count = len(buy_reasons) - strong_buy_count - weak_buy_count
  sell_count = len(sell_reasons)
  short_buy_count = sum(1 for r in buy_reasons if "Short RSI" in r or "Short MACD" in r or "SMA5 > SMA10" in r or "Close > VWAP" in r)
  short_sell_count = sum(1 for r in sell_reasons if "Short RSI" in r or "Short MACD" in r or "SMA5 < SMA10" in r or "Close < VWAP" in r)

  buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * (weak_buy_count + other_buy_count) + 
                    obv_weight * ("OBV Increasing" in buy_reasons) + 
                    bb_width_weight * (f"BB Width < {optimal_params['bb_width_low']}" in buy_reasons) + 
                    w_short_buy * short_buy_count) * 0.1
  sell_adjustment = (w_sell * sell_count + obv_weight * ("OBV Decreasing" in sell_reasons) + 
                     bb_width_weight * (f"BB Width > {optimal_params['bb_width_high']}" in sell_reasons) + 
                     w_short_sell * short_sell_count) * 0.1
  ```
- **목표 비중 계산**:
  - 현재 TSLL 비중(`base_weight`)에 매수 조정값을 더하고 매도 조정값을 뺍니다.
  - 결과는 0%에서 100%로 제한됩니다.
  ```python
  target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))
  ```

#### **4.4. 조정 이유 기록**
- 활성화된 매수/매도 조건을 `reasons` 리스트에 저장하여 사용자에게 제공합니다.

---

## **5. 목표 TSLA 비중 계산**

TSLA의 목표 비중은 TSLL 비중을 기반으로 계산됩니다.
```python
target_tsla_weight = 1 - target_tsll_weight
```

---

## **6. 필요한 주식 수 계산**

`adjust_portfolio` 함수에서 목표 비중에 맞는 주식 수를 계산합니다.
```python
target_tsla_shares = int((target_tsla_weight * total_value) / tsla_close)
target_tsll_shares = int((target_tsll_weight * total_value) / tsll_close)
```

---

## **7. 포트폴리오 조정 제안**

현재 보유 주식 수와 목표 주식 수의 차이를 계산하여 제안합니다.
```python
tsla_diff = target_tsla_shares - current_tsla_shares
tsll_diff = target_tsll_shares - current_tsll_shares
if tsla_diff != 0:
    print(f" - TSLA: {'Buy' if tsla_diff > 0 else 'Sell'} {abs(tsla_diff)} shares (target weight {target_tsla_weight*100:.2f}%)")
if tsll_diff != 0:
    print(f" - TSLL: {'Buy' if tsll_diff > 0 else 'Sell'} {abs(tsll_diff)} shares (target weight {target_tsll_weight*100:.2f}%)")
```

---

## **8. 조정 이유 제공**

`get_target_tsll_weight`에서 생성된 `reasons`를 출력합니다.
```python
print("\n### Adjustment Reasons")
for reason in reasons:
    print(reason)
```

---

## **개선된 점**

1. **단기 지표 도입**: SMA5, SMA10, RSI5, 단기 MACD, VWAP를 추가하여 단기 시장 변동에 민감하게 대응합니다.
2. **동적 파라미터**: VIX에 따라 기본값을 조정하여 시장 변동성에 적응합니다.
3. **정밀한 조정**: 주식 수 단위로 제안하며, 소액 조정도 허용하여 정확도를 높였습니다.
4. **이유 상세화**: 매수/매도 신호를 구체적으로 분류하여 사용자에게 명확한 근거를 제공합니다.

---

## **전체 흐름 요약**

1. **시장 지표 수집**: 다양한 기술적/심리적 지표를 분석합니다.
2. **동적 임계값 설정**: ATR로 변동성에 맞춘 임계값을 조정합니다.
3. **신호 평가**: 매수/매도 조건을 확인하고 가중치를 적용합니다.
4. **비중 계산**: TSLL 목표 비중을 조정하고, TSLA 비중을 설정합니다.
5. **조정 제안**: 필요한 매수/매도 주식 수를 계산합니다.
6. **근거 제공**: 조정 이유를 상세히 출력합니다.

이 로직은 시장 상황을 종합적으로 분석하여 포트폴리오의 수익률을 극대화하려는 전략을 구현합니다.

---

## **Disclaimer**
이 프로그램은 교육 목적으로 제공되며, 투자 조언이 아닙니다. 투자 결정을 내리기 전 금융 전문가와 상담하세요.
