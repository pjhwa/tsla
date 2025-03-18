#!/usr/bin/python3
import requests
from datetime import datetime, timedelta
import pandas as pd

# Usage: fear-greed.py
# 과거 2년간의 Fear & Greed Index 데이터를 수집하여 fear_greed_2years.csv 파일로 저장

# 현재 날짜를 기준으로 설정
today = datetime.now()
start_date = today - timedelta(days=730)
start_date_str = start_date.strftime('%Y-%m-%d')

# URL에서 start_date_str 사용
url = f'https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date_str}'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    historical_data = data['fear_and_greed_historical']['data']
    df = pd.json_normalize(historical_data)
    df['date'] = pd.to_datetime(df['x'], unit='ms')
    df.set_index('date', inplace=True)
    df.to_csv('fear_greed_2years.csv')
    print("데이터가 fear_greed_2years.csv로 저장되었습니다.")
else:
    print(f"요청 실패: 상태 코드 {response.status_code}")
    print(response.text)
