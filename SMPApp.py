import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 


model = load_model('C:\\Users\\mayan\\Machine-learning-projects\\Stock Market Prediction\\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

end = date.today().strftime('%Y-%m-%d')
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
start = st.sidebar.text_input('Start Date', '2013-01-01')
end1 = st.sidebar.text_input('End Date',end)

data = yf.download(stock, start, end)

fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

fig.update_layout(
    title=stock,
    xaxis_title='Date',
    yaxis_title='Price',
    )

st.plotly_chart(fig)


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index= True)
data_test_scale = scaler.fit_transform(data_test)

pricing_data, fundamental_data, moving_averages, Prediction ,news  = st.tabs(['Data Price', 'Fundamental Data','Moving Averages', 'Predictions', 'Top 10 News' ])

with pricing_data:
    st.subheader('Stock Data')
    data1 = data
    data1['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data1.dropna(inplace = True)
    st.write(data1)
    annual_return = data1['% Change'].mean()*252*100
    st.write('Annual Return is ', annual_return,'%')
    stdev = np.std(data1['% Change'])*np.sqrt(252)
    st.write('Standard Deviation', stdev*100,'%')
    st.write('Risk Adj. Return is', annual_return/(stdev*100))


with moving_averages:
    SMA_50, SMA_100, SMA_200, SMA_50vs100, SMA_50vs200, SMA_100vs200 =  st.tabs(['50 Days', '100 Days', '200 Days', '50 Days vs 100 Days', '50 Days vs 200 Days', '100 Days vs 200 days'])
    
    with SMA_50:
        st.subheader('Price vs MA50')
        ma_50_days = data.Close.rolling(50).mean()
        fig1= plt.figure(figsize=(8,6))
        plt.plot(ma_50_days, 'r', label= 'MA50')
        plt.plot(data.Close, 'g')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
        st.plotly_chart(fig1)

    with SMA_100:
        st.subheader('Price vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'b', label='MA100')
        plt.plot(data.Close, 'g')
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.legend()
        plt.show()
        st.plotly_chart(fig2)

    with SMA_200:
        st.subheader('Price vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = plt.figure(figsize=(8,6))
        plt.plot(ma_200_days, 'm', label= 'MA200')
        plt.plot(data.Close, 'g')
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.legend()
        plt.show()
        st.plotly_chart(fig3)

    with SMA_50vs100:
        st.subheader('Price vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig4 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(ma_100_days, 'b', label='MA100')
        plt.plot(data.Close, 'g')
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.legend()
        plt.show()
        st.plotly_chart(fig4)

    with SMA_50vs200:
        st.subheader('Price vs MA50 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig5 = plt.figure(figsize=(8,6))
        plt.plot(ma_50_days, 'r', label= 'MA50')
        plt.plot(ma_200_days, 'm', label= 'MA200')
        plt.plot(data.Close, 'g')
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.legend()
        plt.show()
        st.plotly_chart(fig5)
    
    with SMA_100vs200: 
        st.subheader('Price vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig6 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'b', label='MA100')
        plt.plot(ma_200_days, 'm', label='MA200')
        plt.plot(data.Close, 'g')
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.legend()
        plt.show()
        st.plotly_chart(fig6)
        


with Prediction:
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x),  np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(predict))

    last_observed_price = data['Adj Close'].iloc[-1]

    predict_adjusted = predict + (last_observed_price - predict[0])

    st.subheader('Original Price vs Predicted Price')
    fig7 = plt.figure(figsize=(8,6))
    plt.plot(future_dates, predict_adjusted, 'r', label='Predicted Price')
    plt.plot(data.index, data['Adj Close'], 'g', label='Original Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend()
    plt.show()
    st.plotly_chart(fig7)

    st.subheader('Final Predicted Price')
    final_predicted_price = predict_adjusted[-1]
    st.write(f"The final predicted price is: {final_predicted_price}")

from stocknews import StockNews
with news:
    st.header(f'News {stock}')
    sn = StockNews(stock, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

from alpha_vantage.fundamentaldata import FundamentalData 
with fundamental_data:
    key = 'STMNEZIFSBVAWTNV'
    fd = FundamentalData(key, output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(stock)[0]
    bs = balance_sheet.T[2:]
    bs.colums = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(stock)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(stock)[0]
    cf = cash_flow.T[2:]
    cf.colums = list(cash_flow.T.iloc[0])
    st.write(cf)
