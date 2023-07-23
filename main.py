import plotly.graph_objects as go
import asyncio
import yfinance as yf
import numpy as np
import streamlit as st

async def get_stock_price_history(ticker = 'eurusd=x', period = '1d', interval = '1m', start = None, end = None, actions = True, auto_adjust = True, back_adjust = False, time_to_sleep = 5, current_price = False):
    tesla = yf.Ticker(ticker)
    hist = tesla.history(period=period, interval=interval, start=start, end=end, actions=actions, auto_adjust=auto_adjust, back_adjust=back_adjust)
    current_price = hist['Close'].iloc[-1]
    
    if time_to_sleep != 'dead':
        await asyncio.sleep(time_to_sleep)
        if current_price:
            return hist, current_price
        else:
            return hist
    else:
        if current_price:
            return hist, current_price
        else:
            return hist
            
async def update_values(time_to_sleep = 5):
    hist = await get_stock_price_history(time_to_sleep=time_to_sleep)
    return hist
    
def plot(hist, container = None):
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'])])
    # title is ticker
    fig.update_layout(title='EUR/USD')
    if container is None:
        st.plotly_chart(fig)
    else:
        container.plotly_chart(fig)

if __name__ == '__main__':
    chart_box = st.empty()
    price_box = st.sidebar.empty()

    hist, current_price = asyncio.run(get_stock_price_history(time_to_sleep='dead'))
    while True:
        plot(hist, chart_box)
        message_ = 'Current price: ' + str(current_price) + 'at ' + str(hist.index[-1])
        price_box.subheader(message_)
        hist, current_price = asyncio.run(update_values(time_to_sleep=60))