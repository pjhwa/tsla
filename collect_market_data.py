import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import argparse

# Usage:
# python3 collect_market_data.py
# 기본적으로 TSLA, TSLL 주가 데이터를 수집.
# python3 script.py --tickers AAPL,GOOGL
# AAPL, GOOGL 주가 데이터를 수집.
#
# Output:
# fear_greed_2years.csv: Fear & Greed Index 데이터.
# TSLA-history-2y.csv, TSLL-history-2y.csv (또는 지정된 티커): 주식 데이터.
# VIX-history-2y.csv: VIX 지수 데이터.
  
# Fear & Greed Index 데이터 수집 함수
def get_fear_greed_data():
    today = datetime.now()
    start_date = today - timedelta(days=730)
    start_date_str = start_date.strftime('%Y-%m-%d')
    url = f'https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date_str}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 상태 코드가 200이 아니면 예외 발생
        data = response.json()
        historical_data = data['fear_and_greed_historical']['data']
        df = pd.json_normalize(historical_data)
        df['date'] = pd.to_datetime(df['x'], unit='ms')
        df.set_index('date', inplace=True)
        df.to_csv('fear_greed_2years.csv')
        print("Fear & Greed Index 데이터가 fear_greed_2years.csv로 저장되었습니다.")
    except Exception as e:
        print(f"Fear & Greed Index 데이터 수집 오류: {e}")

# 주식 데이터 수집 함수
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='2y')
        hist = hist.reset_index()
        df = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
        df['Open'] = df['Open'].round(2)
        df['High'] = df['High'].round(2)
        df['Low'] = df['Low'].round(2)
        df['Close'] = df['Close'].round(2)
        df['Volume'] = df['Volume'].apply(lambda x: f'{x:,}')
        filename = f'{ticker}-history-2y.csv'
        df.to_csv(filename, index=False, quoting=1)
        print(f"{ticker} 데이터가 {filename}에 저장되었습니다.")
    except Exception as e:
        print(f"{ticker} 데이터 수집 오류: {e}")

# VIX 지수 데이터 수집 함수
def get_vix_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    try:
        vix_df = yf.download("^VIX", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        vix_df = vix_df[['Close']]
        vix_df.reset_index(inplace=True)
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        vix_df.set_index('Date', inplace=True)
        vix_df.to_csv("VIX-history-2y.csv")
        print("VIX 지수 데이터가 VIX-history-2y.csv로 저장되었습니다.")
    except Exception as e:
        print(f"VIX 지수 데이터 수집 오류: {e}")

# 메인 함수
def main():
    # 명령줄 인수 처리
    parser = argparse.ArgumentParser(description='과거 2년간의 데이터를 수집하여 CSV 파일로 저장')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='주식 티커 (쉼표로 구분, 기본: TSLA,TSLL)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    # 데이터 수집 실행
    print("데이터 수집을 시작합니다...")
    get_fear_greed_data()  # Fear & Greed Index 데이터 수집
    for ticker in tickers:  # 주식 데이터 수집
        get_stock_data(ticker)
    get_vix_data()  # VIX 지수 데이터 수집
    print("모든 데이터 수집이 완료되었습니다.")

if __name__ == "__main__":
    main()
