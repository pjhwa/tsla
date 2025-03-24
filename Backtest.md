# Portfolio Optimization using Backtesting

이 프로젝트는 과거 주식 데이터를 활용하여 포트폴리오를 최적화하는 백테스팅 도구입니다. TSLA와 TSLL 주식 데이터를 기반으로 기술적 지표를 계산하고, Grid Search, 유전 알고리즘(GA), 베이지안 최적화(Bayesian Optimization) 등의 방법을 통해 최적의 매수/매도 파라미터를 탐색합니다. 멀티프로세싱을 지원하여 계산 성능을 극대화하며, 결과를 로그 파일과 JSON 형식으로 저장합니다.

## Installation

### 1. 필요한 라이브러리 설치

아래 명령어를 사용하여 필요한 Python 라이브러리를 설치하세요.

```bash
pip install pandas numpy deap bayesian_optimization tqdm
```

### 2. 데이터 파일 준비

다음 세 개의 CSV 파일이 필요합니다.
- `fear_greed_2years.csv`: Fear & Greed Index 데이터.
- `TSLA-history-2y.csv`: TSLA 주가 데이터.
- `TSLL-history-2y.csv`: TSLL 주가 데이터.
파일은 프로젝트 디렉토리에 저장해야 하며, 자세한 형식은 **Requirements** 섹션을 참조하세요.

## Requirements

### `fear_greed_2years.csv`
- **설명**: Fear & Greed Index 데이터.
- **필수 열**:
  - `date`: 날짜 (형식: `%m/%d/%Y`).
  - `y`: Fear & Greed Index 값 (숫자).
- **생성 방법**
  ```bash
  python3 fear_greed.py
  ```
  
### `TSLA-history-2y.csv`
- **설명**: TSLA 주가 데이터.
- **필수 열**:
  - `Date`: 날짜 (형식: `%m/%d/%Y`).
  - `Close`: 종가.
  - `High`: 고가.
  - `Low`: 저가.
  - `Volume`: 거래량 (쉼표 포함 문자열, 예: "1,234,567").
- **생성 방법**
  ```bash
  python3 getdata.py --ticker=TSLA
  ```
  
### `TSLL-history-2y.csv`
- **설명**: TSLL 주가 데이터.
- **필수 열**: 위와 동일 (`TSLL_` 접두사가 내부적으로 추가됨).
- **생성 방법**
  ```bash
  python3 getdata.py --ticker=TSLL
  ```
  
### 데이터 전처리
- 날짜는 `parse_dates`와 `date_format='%m/%d/%Y'`로 처리됩니다.
- `Volume` 열의 쉼표는 제거되고 float로 변환됩니다.
- 결측치는 제거되며, RSI, SMA, MACD, 볼린저 밴드, ATR 등이 계산됩니다.

## Usage

명령줄에서 최적화 방식을 선택하여 코드를 실행할 수 있습니다. 예시는 아래와 같습니다.

### - Grid Search:
  ```bash
  python backtest.py --method grid_search --n_samples 50000 --processes 4
  ```
  
### - GA (유전 알고리즘):
  ```bash
  python backtest.py --method ga --population_size 200 --generations 100 --processes 4
  ```

### - Bayesian Optimization:
  ```bash
  python backtest.py --method bayesian --init_points 5 --n_iter 20
  ```
  
## 코드 구조 및 기능

### 1. 데이터 로드 및 전처리 (`load_and_prepare_data`)

- Fear & Greed Index, TSLA, TSLL 데이터를 로드하고 병합합니다.
- 거래량 데이터를 숫자 형식으로 변환하고 결측값을 제거합니다.
- 다음과 같은 기술적 지표를 계산합니다.
  - **Daily RSI**: 일일 상대강도지수
  - **Weekly RSI**: 주간 상대강도지수
  - **SMA50, SMA200**: 50일, 200일 단순 이동평균
  - **Bollinger Bands**: 상단/하단 밴드
  - **Volume Change**: 거래량 변화율
  - **MACD**: 이동평균 수렴 발산 (MACD와 신호선)
  - **ATR**: 평균 진폭 범위

### 2. 지표 계산 함수

- `calculate_rsi`: RSI를 계산합니다 (기간 설정 가능).
- `calculate_sma`: 단순 이동평균을 계산합니다.
- `calculate_bollinger_bands`: 볼린저 밴드를 계산합니다 (기본: 20일, 2 표준편차).
- `calculate_macd`: MACD와 신호선을 계산합니다 (기본: 12/26/9).
- `calculate_atr`: 평균 진폭 범위를 계산합니다 (기본: 14일).

### 3. 백테스팅 함수 (`simulate_backtest_vectorized`)

- 주어진 파라미터를 사용하여 매수/매도 조건을 벡터화된 방식으로 설정합니다.
- **매수 조건**:
  - Daily RSI < `daily_rsi_buy`
  - Weekly RSI < `weekly_rsi_buy`
  - 종가 > 하단 Bollinger Band
  - Volume Change > `volume_change_buy`
  - Fear & Greed < `fg_buy`
  - MACD > MACD 신호선
- **매도 조건**: 반대 조건 적용.
- **ATR 기반 가중치 조정**: ATR을 활용해 동적으로 매수/매도 비중을 계산합니다.
- 초기 포트폴리오 가치(100,000)를 기준으로 최종 가치를 계산합니다.

### 4. 최적화 함수

- `optimize_grid_search_parallel`: 병렬 Grid Search로 파라미터 조합을 탐색합니다.
- `optimize_ga_parallel`: 유전 알고리즘을 통해 최적 파라미터를 탐색합니다.
- `optimize_bayesian`: 베이지안 최적화를 사용하여 효율적으로 탐색합니다.

## 최적화 방식

### 1. 그리드 서치 (Grid Search)

- **설명**: 정의된 파라미터 범위에서 랜덤 샘플링을 통해 조합을 평가합니다.
- **특징**: 철저하지만 계산 비용이 높습니다. 병렬 처리를 지원하며, 진행 상황을 `tqdm`으로 시각화합니다.
- **파라미터 그리드**:
  - `daily_rsi_buy`: [10, 20, 30, 40],
  - `daily_rsi_sell`: [60, 70, 80, 90],
  - `weekly_rsi_buy`: [20, 30, 40, 50],
  - `weekly_rsi_sell`: [50, 60, 70, 80],
  - `fg_buy`: [10, 20, 30, 40, 50, 60],
  - `fg_sell`: [60, 70, 80, 90],
  - `volume_change_buy`: [-1.0, -0.5, 0.0, 0.5, 1.0],
  - `volume_change_sell`: [-1.0, -0.5, 0.0, 0.5, 1.0],
  - `w_buy`: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
  - `w_sell`: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- **총 조합 수**: 625,000개 (5^10).
- **관련 인자**:
  - `--n_samples`: 탐색할 샘플 조합 수 (기본: 100,000)

### 2. 유전 알고리즘 (Genetic Algorithm)

- **설명**: 진화론적 접근으로 개체군을 생성하고 교차 및 돌연변이를 통해 최적해를 탐색하며, 병렬 처리가 가능합니다.
- **특징**: 효율적이며 복잡한 문제에 적합합니다.
- **파라미터 범위**:
  - `daily_rsi_buy`: (10, 40)
  - `daily_rsi_sell`: (60, 90)
  - `weekly_rsi_buy`: (20, 50)
  - `weekly_rsi_sell`: (50, 80)
  - `fg_buy`: (10, 60)
  - `fg_sell`: (60, 90)
  - `volume_change_buy`: (-1.0, 1.0)
  - `volume_change_sell`: (-1.0, -1.0)
  - `w_buy`: (0.5, 3.0)
  - `w_sell`: (0.5, 3.0)
- **관련 인자**:
  - `--population_size`: 개체군 크기 (기본: 100)
  - `--generations`: 세대 수 (기본: 50)

### 3. 베이지안 최적화 (Bayesian Optimization)

- **설명**: 확률 모델을 활용해 효율적으로 최적 파라미터를 탐색합니다.
- **특징**: 초기 탐색 후 반복적으로 탐색 공간을 좁혀갑니다. 계산 비용이 낮고 빠른 수렴이 가능합니다.
- **파라미터 범위**: 유전 알고리즘과 동일.
- **관련 인자**:
  - `--init_points`: 초기 탐색 포인트 수 (기본: 10)
  - `--n_iter`: 반복 횟수 (기본: 50)

## 명령줄 인자 및 사용 예시

### 공통 인자
- `--method`: 최적화 방식 (`grid_search`, `ga`, `bayesian`)
- `--processes`: 병렬 처리에 사용할 CPU 코어 수 (기본: 시스템 코어 수)

### Grid Search 예시
```bash
python backtest.py --method grid_search --n_samples 50000 --processes 4
```

### GA 예시
```bash
python backtest.py --method ga --population_size 200 --generations 100 --processes 4
```

### Bayesian Optimization 예시
```bash
python backtest.py --method bayesian --init_points 5 --n_iter 20
```

## 출력 및 결과 해석

### 출력 예시

- **그리드 서치**:
  ```
  그리드 서치 시작: 총 625000개의 조합
  Grid Search Progress:   1%|▏         | 6250/625000 [00:10<16:40, 625.00it/s]
  진행 중: 6250/625000, 예상 남은 시간: 1000.00초
  ...
  그리드 서치 완료: 최적 파라미터 {'daily_rsi_buy': 30, ...}, 최종 포트폴리오 가치 150000.00
  ```
  
- **유전 알고리즘**:
  ```
  유전 알고리즘 시작: 40세대
  GA Progress:   2%|▎         | 1/40 [00:05<03:15,  5.00s/it]
  세대 1/40 완료, 소요 시간: 5.00초
  ...
  유전 알고리즘 완료: 최적 파라미터 {'daily_rsi_buy': 28.5, ...}, 최종 포트폴리오 가치 145000.00
  ```

- **베이지안 최적화**:
  ```
  베이지안 최적화 시작: 초기 포인트 5개, 반복 25회
  |   iter    |  target   | daily_... | ...
  ...
  베이지안 최적화 완료, 총 소요 시간: 120.00초
  최적 파라미터: {'daily_rsi_buy': 32.1, ...}, 최종 포트폴리오 가치: 148000.00
  ```

### 결과 해석
- **최적 파라미터**: 매수/매도 조건과 비중 조정 가중치.
- **포트폴리오 가치**: 초기 $100,000에서 최종 가치.
- **수익률**: 백테스트 기간 동안의 수익률(%).

## 로그 파일 및 결과 저장

- 로그 파일: `optimization_log`에 실행 기록이 추가됩니다 (시간, 상태, 메시지 포함).
- 최적 파라미터: `optimal_params.json`에 JSON 형식으로 저장됩니다.
  - 예: `{"daily_rsi_buy": 30, "daily_rsi_sell": 70, ...}`

### 주의사항

- 데이터 파일의 경로와 형식이 코드와 일치해야 합니다.
- Grid Search에서 샘플 수가 많을 경우 실행 시간이 길어질 수 있습니다.
- 대규모 병렬 처리를 위해 충분한 메모리와 CPU 자원이 필요합니다.

## Disclaimer
이 프로그램은 교육 목적으로만 제공되며, 투자 조언으로 간주되지 않습니다. 투자 결정을 내리기 전에 반드시 금융 전문가와 상담하시기 바랍니다.
