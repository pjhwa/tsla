#!/usr/bin/python3
import yfinance as yf
import pandas as pd
import argparse

# Usage: getdata.py --ticker TSLA
# 주어진 티커의 과거 2년간의 데이터를 수집하여 {Ticker}-history-2y.csv 파일로 저장

# 명령줄 인수 처리
parser = argparse.ArgumentParser(description='주식 과거 데이터를 CSV 파일로 저장')
parser.add_argument('--ticker', type=str, default='TSLA', help='주식 티커 (기본: TSLA)')
args = parser.parse_args()

# 지정된 티커로 데이터 수집
ticker = args.ticker
stock = yf.Ticker(ticker)
hist = stock.history(period='2y')

# DataFrame을 리셋하고 필요한 컬럼만 선택 (명시적 복사본 생성)
hist = hist.reset_index()
df = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Date를 'MM/DD/YYYY' 형식으로 변환
df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')

# 주가 데이터를 소수점 둘째 자리까지 반올림
df['Open'] = df['Open'].round(2)
df['High'] = df['High'].round(2)
df['Low'] = df['Low'].round(2)
df['Close'] = df['Close'].round(2)

# Volume을 쉼표가 포함된 문자열로 변환
df['Volume'] = df['Volume'].apply(lambda x: f'{x:,}')

# CSV 파일로 저장
filename = f'{ticker}-history-2y.csv'
df.to_csv(filename, index=False, quoting=1)  # 모든 필드를 따옴표로 감쌈

print(f"데이터가 {filename} 파일에 저장되었습니다.")
