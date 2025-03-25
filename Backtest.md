# Portfolio Backtesting with Genetic Algorithm (`backtest.py`)

## Overview

`backtest.py`는 유전 알고리즘(Genetic Algorithm, GA)을 활용하여 포트폴리오 조정 전략의 최적 파라미터를 도출하는 Python 스크립트입니다. 이 스크립트는 과거 주가 데이터(TSLA 및 TSLL)와 공포탐욕지수(Fear & Greed Index)를 기반으로 백테스트를 수행하여 포트폴리오의 수익률을 극대화하는 파라미터를 찾습니다. 최적화된 파라미터는 `optimal_params.json` 파일로 저장됩니다.

## Features

- **유전 알고리즘(GA)**: DEAP 라이브러리를 사용하여 최적 파라미터 탐색.
- **진행률 표시**: `tqdm`을 통해 백테스트와 GA 진행 상황을 시각적으로 확인.
- **기술적 지표 계산**: RSI, SMA, MACD, 볼린저 밴드, ATR, 주간 RSI 등 다양한 지표 활용.
- **파라미터 최적화**: 공포탐욕지수, RSI, 거래량 변화율 등의 매수/매도 임계값 및 가중치 최적화.

## Requirements

- **Python**: 3.6 이상
- **필요한 라이브러리**:
  - `pandas`
  - `numpy`
  - `deap`
  - `scipy`
  - `tqdm`

### Installation

1. **필수 라이브러리 설치**:
   ```bash
   pip install pandas numpy deap scipy tqdm
   ```

2. **데이터 파일 준비**:
   스크립트와 동일한 디렉토리에 아래 파일들이 필요합니다:
   - `fear_greed_2years.csv`: 공포탐욕지수 데이터 (컬럼: `date`, `y`, 날짜 형식: `YYYY-MM-DD`)
   - `TSLA-history-2y.csv`: TSLA 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)
   - `TSLL-history-2y.csv`: TSLL 주가 데이터 (컬럼: `Date`, `Close`, `High`, `Low`, `Volume`, 날짜 형식: `MM/DD/YYYY`)

   **참고**: 데이터 파일은 CSV 형식이어야 하며, 날짜 형식이 코드와 일치해야 합니다.

## Usage

### Running the Script

터미널에서 다음 명령어로 스크립트를 실행하세요.
```bash
python backtest.py
```

- **진행 상황**: `tqdm`을 통해 백테스트와 유전 알고리즘의 진행률이 표시됩니다.
- **출력**: 최적 파라미터와 해당 파라미터로 얻은 최대 수익률이 콘솔에 출력되며, `optimal_params.json` 파일로 저장됩니다.

### Example Output

```
Genetic Algorithm Progress: 100%|██████████| 50/50 [02:30<00:00,  3.00s/it]
최적 파라미터: {
    "fg_buy": 30.5, "fg_sell": 70.2, "daily_rsi_buy": 28.3, "daily_rsi_sell": 72.1,
    "weekly_rsi_buy": 35.7, "weekly_rsi_sell": 65.4, "volume_change_strong_buy": 0.55,
    "volume_change_weak_buy": 0.25, "volume_change_sell": -0.15,
    "w_strong_buy": 1.8, "w_weak_buy": 1.2, "w_sell": 0.9
}
최대 수익률: 25.67%
```

## Code Structure

### Functions

1. **`load_data()`**:
   - **설명**: TSLA와 TSLL 주가 데이터 및 공포탐욕지수를 로드하고 병합합니다. RSI, SMA(50, 200), MACD, 볼린저 밴드, ATR, 주간 RSI 등 기술적 지표를 계산하여 데이터프레임에 추가합니다.
   - **입력**: 없음 (CSV 파일에서 데이터 로드).
   - **출력**: 전처리 및 지표 계산이 완료된 데이터프레임.

2. **지표 계산 함수**:
   - **`calculate_rsi(series, timeperiod)`**: 상대강도지수(RSI)를 계산.
   - **`calculate_sma(series, timeperiod)`**: 단순이동평균(SMA)을 계산.
   - **`calculate_macd(series, fastperiod, slowperiod, signalperiod)`**: MACD와 시그널 라인을 계산.
   - **`calculate_bollinger_bands(series, timeperiod, nbdevup, nbdevdn)`**: 볼린저 밴드(상단, 중간, 하단)를 계산.
   - **`calculate_atr(df, timeperiod)`**: 평균 истин 범위(ATR)를 계산.
   - **`get_rsi_trend(rsi_series, window)`**: RSI의 추세(상승/하락/안정)를 판단.

3. **`evaluate(individual, data)`**:
   - **설명**: 주어진 파라미터(`individual`)로 백테스트를 수행하고 수익률을 계산합니다. `tqdm`으로 진행 상황을 표시.
   - **입력**: 유전 알고리즘의 개체(파라미터 리스트), 데이터프레임.
   - **출력**: 백테스트 결과로 얻은 수익률.

4. **`get_target_tsll_weight(...)`**:
   - **설명**: 공포탐욕지수, RSI, MACD, 볼린저 밴드, 거래량 변화율 등 다양한 지표를 기반으로 TSLL의 목표 비중을 계산합니다. 매수/매도 조건을 평가하여 비중을 조정.
   - **입력**: 현재 시장 데이터 및 파라미터.
   - **출력**: TSLL의 목표 비중.

5. **`main()`**:
   - **설명**: 유전 알고리즘을 설정하고 실행하여 최적 파라미터를 도출합니다. 결과를 출력하고 `optimal_params.json`에 저장.
   - **입력**: 없음.
   - **출력**: 최적 파라미터와 최대 수익률.

### Logic Explanation

- **유전 알고리즘(GA) 설정**:
  - **개체군(population)**: 파라미터 조합을 나타내는 개체들로 구성.
  - **적합도 함수**: `evaluate` 함수를 통해 각 개체의 수익률을 계산.
  - **교차(crossover)**: `cxBlend`를 사용하여 두 개체의 파라미터를 블렌딩.
  - **돌연변이(mutation)**: `mutGaussian`을 사용하여 파라미터에 가우시안 노이즈 추가.
  - **선택(selection)**: `selTournament`를 통해 토너먼트 방식으로 우수한 개체 선택.

- **백테스트 로직**:
  1. **포트폴리오 초기화**: 초기 자산 $100,000, TSLA와 TSLL 주식 보유량 0주.
  2. **일일 조정**:
     - `get_target_tsll_weight`를 통해 TSLL의 목표 비중을 계산.
     - 목표 비중에 따라 TSLA와 TSLL 주식을 매수/매도하여 포트폴리오 조정.
  3. **최종 수익률 계산**: 백테스트 기간 종료 후 수익률 반환.

- **매수/매도 조건**:
  - **매수 조건**:
    - 공포탐욕지수 ≤ `fg_buy`
    - 일일 RSI < `daily_rsi_buy`
    - MACD > 시그널 (음수 구간)
    - 거래량 변화율 > `volume_change_strong_buy` (강한 매수) 또는 > `volume_change_weak_buy` (약한 매수)
    - 주가 < 볼린저 하단 밴드
    - RSI 상승 추세 & 주가 > SMA200
  - **매도 조건**:
    - 공포탐욕지수 ≥ `fg_sell`
    - 일일 RSI > `daily_rsi_sell`
    - MACD < 시그널 (양수 구간)
    - 거래량 변화율 < `volume_change_sell`
    - 주가 > 볼린저 상단 밴드
    - RSI 하락 추세 & 주가 < SMA200

## Genetic Algorithm Parameters

### Population Size and Generation Number

유전 알고리즘의 성능은 **개체군 수(population size)**와 **세대 수(generation number)**에 크게 의존합니다. 이 두 파라미터는 최적화 품질과 계산 비용 간의 균형을 맞추는 데 중요합니다.

- **개체군 수 (Population Size)**:
  - **정의**: 한 세대에서 평가되는 파라미터 조합(개체)의 수.
  - **영향**:
    - **작은 개체군** (예: 20~50): 계산 시간이 적지만, 다양성 부족으로 지역 최적해에 갇힐 가능성 높음.
    - **큰 개체군** (예: 200 이상): 다양성이 증가하여 전역 최적해 탐색 가능성 높아짐, 단 계산 부담 증가.
  - **권장값**: 파라미터 수(12개)의 5~10배, 즉 60~120. 기본 설정은 100.

- **세대 수 (Generation Number)**:
  - **정의**: 유전 알고리즘이 진화 과정을 반복하는 횟수.
  - **영향**:
    - **적은 세대** (예: 20~30): 최적화가 충분히 이루어지지 않아 좋은 해를 찾지 못할 가능성.
    - **많은 세대** (예: 200 이상): 최적해에 근접 가능성 높지만, 과적합 위험 있음.
  - **권장값**: 50~100. 기본 설정은 50.

- **최적 조합**:
  - 개체군 수와 세대 수는 서로 보완적입니다. **개체군 수 × 세대 수**가 5,000~10,000번 평가를 넘으면 최적해 탐색 가능성이 높아집니다.
  - 예: 개체군 100, 세대 50 → 5,000번 평가.
  - 계산 자원에 따라 조정 가능하며, 병렬 처리를 활용하면 실행 시간 단축 가능.

### Other GA Parameters

- **교차 확률 (crossover probability)**: 0.7
- **돌연변이 확률 (mutation probability)**: 0.3
- **토너먼트 크기 (tournament size)**: 3

## Disclaimer

이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
