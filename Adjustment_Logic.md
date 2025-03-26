## **비중 조정 로직의 개요**
`portfolio_pulse.py`는 TSLA(테슬라 주식)와 TSLL(테슬라 레버리지 ETF)의 포트폴리오 비중을 시장 상황에 따라 동적으로 조정하는 프로그램입니다. 주요 목표는 시장 지표를 분석하여 TSLL의 목표 비중을 계산하고, 이를 현재 비중과 비교해 매수 또는 매도 제안을 제공하는 것입니다. 이 과정은 다음과 같은 단계로 진행됩니다.

1. **시장 지표 수집 및 분석**
2. **최적 파라미터 로드**
3. **현재 포트폴리오 상태 파악**
4. **목표 TSLL 비중 계산**
5. **필요한 주식 수 계산**
6. **포트폴리오 조정 제안**
7. **결과 로깅**

이제 각 단계를 코드와 함께 상세히 살펴보겠습니다.

---

## **1. 시장 지표 수집 및 분석**
시장 상황을 평가하기 위해 다양한 기술적 지표와 심리 지표를 수집합니다. 이 지표들은 TSLL 비중 조정의 핵심 입력값으로 사용됩니다.

### **수집되는 지표**
- **Fear & Greed Index**: CNN에서 제공하는 시장 심리 지표로, 공포(0)에서 탐욕(100)까지의 값을 가집니다. `get_fear_greed_index` 함수에서 최신 값을 가져옵니다.
- **RSI (Relative Strength Index)**: 
  - 일일 RSI는 `get_stock_data`에서 TSLA의 14일 RSI를 계산합니다.
  - 주간 RSI는 `get_weekly_rsi`에서 TSLA의 주간 데이터를 기반으로 계산합니다.
  - 과매수(70 이상) 또는 과매도(30 이하) 상태를 판단합니다.
- **SMA (Simple Moving Average)**: TSLA의 50일 및 200일 단순 이동평균을 `calculate_sma`로 계산하여 단기 및 장기 추세를 분석합니다.
- **Bollinger Bands**: `calculate_bollinger_bands`로 20일 SMA와 상단/하단 밴드(표준편차 2배)를 계산하여 가격 변동성을 측정합니다.
- **MACD (Moving Average Convergence Divergence)**: `calculate_macd`로 12일/26일 EMA 차이와 9일 신호선을 계산하여 추세 전환을 포착합니다.
- **Volume Change**: 거래량 변화율(`pct_change`)을 계산하여 시장 참여자의 관심도를 파악합니다.
- **ATR (Average True Range)**: `calculate_atr`로 14일 평균 진폭을 계산하여 가격 변동성을 반영합니다.

### **코드 예시**
```python
def get_stock_data(ticker, period="max", interval="1d"):
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df['RSI'] = calculate_rsi(df['Close'], timeperiod=14)
    df['SMA50'] = calculate_sma(df['Close'], timeperiod=50)
    df['SMA200'] = calculate_sma(df['Close'], timeperiod=200)
    df['Upper Band'], df['Middle Band'], df['Lower Band'] = calculate_bollinger_bands(df['Close'])
    df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
    df['Volume Change'] = df['Volume'].pct_change()
    df['ATR'] = calculate_atr(df, timeperiod=14)
    return df
```
- TSLA와 TSLL의 데이터를 `yfinance`로 가져와 각 지표를 계산합니다.
- 결측치는 선형 보간(`interpolate`)으로 처리됩니다.

---

## **2. 최적 파라미터 로드**
비중 조정의 임계값과 가중치는 `optimal_params.json` 파일에서 로드됩니다. 이 값들은 백테스트로 최적화된 설정으로, 없으면 기본값이 사용됩니다.

### **파라미터 예시**
```python
def load_optimal_params(file_path="optimal_params.json"):
    try:
        with open(file_path, "r") as f:
            return json.loads(f.read())
    except:
        return {
            "fg_buy": 25,  # Fear & Greed 매수 임계값
            "fg_sell": 75,  # Fear & Greed 매도 임계값
            "daily_rsi_buy": 30,  # 일일 RSI 매수 임계값
            "daily_rsi_sell": 70,  # 일일 RSI 매도 임계값
            "volume_change_strong_buy": 0.5,  # 강한 매수 거래량 변화율
            "w_strong_buy": 2.0,  # 강한 매수 가중치
            "w_weak_buy": 1.0,  # 약한 매수 가중치
            "w_sell": 1.0  # 매도 가중치
        }
```
- **임계값**: 매수/매도 신호를 판단하는 기준.
- **가중치**: 신호의 강도에 따라 비중 조정에 미치는 영향을 조절.

---

## **3. 현재 포트폴리오 상태 파악**
현재 TSLA와 TSLL의 보유 주식 수와 비중을 확인합니다.

### **코드 동작**
- **`load_transactions`**: `transactions.txt` 파일에서 거래 내역을 읽어 보유 주식 수를 계산합니다.
  ```python
  def load_transactions(file_path="transactions.txt"):
      if not os.path.exists(file_path):
          return {}, 0
      transactions_df = pd.read_csv(file_path, sep='\s+', names=["date", "ticker", "action", "shares", "stock_price"])
      holdings = {}
      for _, row in transactions_df.iterrows():
          ticker = row['ticker']
          if row['action'] == "buy":
              holdings[ticker] = holdings.get(ticker, 0) + row['shares']
          elif row['action'] == "sell":
              holdings[ticker] = holdings.get(ticker, 0) - row['shares']
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
- 파일이 없으면 초기 자산(`$100,000`)을 기준으로 비중을 계산합니다.

---

## **4. 목표 TSLL 비중 계산**
`get_target_tsll_weight` 함수는 시장 지표와 최적 파라미터를 활용해 TSLL의 목표 비중을 계산합니다. 이 과정이 비중 조정의 핵심입니다.

### **세부 단계**
#### **4.1. 동적 임계값 계산**
- ATR을 사용해 변동성에 따라 거래량 변화율 임계값을 동적으로 조정합니다.
  ```python
  atr_normalized = atr / close
  volume_change_strong_buy = optimal_params["volume_change_strong_buy"] * (1 + atr_normalized)
  volume_change_weak_buy = optimal_params["volume_change_weak_buy"] * (1 + atr_normalized)
  volume_change_sell = optimal_params["volume_change_sell"] * (1 + atr_normalized)
  ```
- 변동성이 클수록 더 큰 거래량 변화를 요구합니다.

#### **4.2. 매수 및 매도 조건 평가**
- **매수 조건**:
  - Fear & Greed Index ≤ `fg_buy` (예: 25)
  - 일일 RSI < `daily_rsi_buy` (예: 30)
  - MACD > 신호선 & 신호선 < 0
  - 거래량 변화율 > `volume_change_strong_buy` (강한 매수) 또는 `volume_change_weak_buy` (약한 매수)
  - 종가 < 하단 Bollinger Band
  - RSI 상승 & 종가 > 200일 SMA
- **매도 조건**:
  - Fear & Greed Index ≥ `fg_sell` (예: 75)
  - 일일 RSI > `daily_rsi_sell` (예: 70)
  - MACD < 신호선 & 신호선 > 0
  - 거래량 변화율 < `volume_change_sell`
  - 종가 > 상단 Bollinger Band
  - RSI 하락 & 종가 < 200일 SMA
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
  buy_reasons = [condition for condition, is_true in buy_conditions.items() if is_true]
  sell_reasons = [condition for condition, is_true in sell_conditions.items() if is_true]
  ```

#### **4.3. 가중치 적용 및 비중 조정**
- **가중치 계산**:
  - "강한 매수" 신호: `w_strong_buy` (예: 2.0)
  - "약한 매수" 및 기타 매수 신호: `w_weak_buy` (예: 1.0)
  - 매도 신호: `w_sell` (예: 1.0)
  ```python
  strong_buy_count = sum(1 for r in buy_reasons if "Strong Buy" in r)
  weak_buy_count = sum(1 for r in buy_reasons if "Weak Buy" in r and "Strong Buy" not in r)
  other_buy_count = len(buy_reasons) - strong_buy_count - weak_buy_count
  sell_count = len(sell_reasons)
  buy_adjustment = (w_strong_buy * strong_buy_count + w_weak_buy * weak_buy_count + w_weak_buy * other_buy_count) * 0.1
  sell_adjustment = w_sell * sell_count * 0.1
  ```
- **목표 비중 계산**:
  - 현재 비중(`base_weight`)에 매수 조정값을 더하고 매도 조정값을 뺍니다.
  - 결과는 0%~100%로 제한됩니다.
  ```python
  target_weight = max(0.0, min(base_weight + buy_adjustment - sell_adjustment, 1.0))
  ```

#### **4.4. 조정 이유 기록**
- 활성화된 매수/매도 조건을 `reasons` 리스트에 저장하여 사용자에게 제공합니다.

---

## **5. 필요한 주식 수 계산**
`calculate_required_shares` 함수는 목표 비중에 맞는 TSLA와 TSLL 주식 수를 계산합니다.
```python
def calculate_required_shares(target_tsla_weight, target_tsll_weight, total_value, tsla_close, tsll_close):
    target_tsla_value = target_tsla_weight * total_value
    target_tsll_value = target_tsll_weight * total_value
    required_tsla_shares = int(target_tsla_value / tsla_close)
    required_tsll_shares = int(target_tsll_value / tsll_close)
    return required_tsla_shares, required_tsll_shares
```
- TSLA 비중 = `1 - target_tsll_weight`.

---

## **6. 포트폴리오 조정 제안**
`adjust_portfolio` 함수는 현재와 목표 비중의 차이를 계산하여 매수/매도 제안을 출력합니다.
```python
def adjust_portfolio(target_tsll_weight, current_tsll_weight, total_value, tsll_close):
    target_tsll_value = target_tsll_weight * total_value
    current_tsll_value = current_tsll_weight * total_value
    difference = target_tsll_value - current_tsll_value
    shares_to_adjust = int(difference / tsll_close)
    if abs(difference) < 100:
        print(" - No significant adjustment needed.")
    elif difference > 0:
        print(f" - Buy TSLL: ${difference:.2f} (approx. {shares_to_adjust} shares)")
    else:
        print(f" - Sell TSLL: ${-difference:.2f} (approx. {-shares_to_adjust} shares)")
```
- 차이가 $100 미만이면 조정 불필요로 판단합니다.

---

## **7. 결과 로깅**
`log_decision` 함수는 조정 내역을 `portfolio_log.csv`에 기록합니다.
```python
def log_decision(date, target_tsll_weight, reasons):
    with open("portfolio_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date, target_tsll_weight, "; ".join(reasons)])
```

---

## **전체 흐름 요약**
1. **시장 지표 분석**: Fear & Greed Index, RSI, SMA, Bollinger Bands, MACD, 거래량 변화율, ATR을 수집합니다.
2. **동적 임계값**: ATR로 변동성에 맞춘 거래량 임계값을 설정합니다.
3. **매수/매도 신호**: 지표를 기반으로 조건을 평가하고 가중치를 적용합니다.
4. **비중 조정**: 현재 TSLL 비중에 매수/매도 조정값을 반영해 목표 비중을 계산합니다.
5. **조정 제안**: 목표 비중에 맞춰 TSLL의 매수/매도 수량을 제안합니다.

이 로직은 시장 상황을 종합적으로 분석하여 TSLL 비중을 동적으로 조정하며, TSLA와의 균형을 유지해 수익을 극대화하려는 전략을 구현합니다.
