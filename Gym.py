''' 
The idea is that we give the model the minimum amount of information and let it figure out the rest.
- take the data for the month of January

'''
import pandas as pd
import yfinance as yf
import streamlit as st
st.set_page_config(layout="wide")
import plotly.graph_objects as go
import time
import random
import numpy as np
from plotly.subplots import make_subplots


class Analyser:
    def __init__(self):
        pass

    def MACD(self,hist):
        # MACD
        exp1 = hist.Close.ewm(span=12, adjust=False).mean()
        exp2 = hist.Close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist['MACD'] = macd
        hist['Signal'] = signal
        hist['Histogram'] = hist['MACD'] - hist['Signal']
        return hist
    
    def RSI(self,hist):
        delta = hist['Close'].diff()
        up_days = delta.copy()
        up_days[delta<=0]=0.0
        down_days = abs(delta.copy())
        down_days[delta>0]=0.0
        RS_up = up_days.ewm(com=13, adjust=False).mean()
        RS_down = down_days.ewm(com=13, adjust=False).mean()
        RS = RS_up/RS_down
        RSI = 100.0 - (100.0/(1.0+RS))
        hist['RSI'] = RSI
        return hist
    
    def BollingerBands(self,hist):
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['20dSTD'] = hist['Close'].rolling(20).std(ddof=0)
        hist['Upper'] = hist['MA20'] + (hist['20dSTD'] * 2)
        hist['Lower'] = hist['MA20'] - (hist['20dSTD'] * 2)
        return hist
    
    def ATR(self,hist):
        hist['H-L']=abs(hist['High']-hist['Low'])
        hist['H-PC']=abs(hist['High']-hist['Close'].shift(1))
        hist['L-PC']=abs(hist['Low']-hist['Close'].shift(1))
        hist['TR']=hist[['H-L','H-PC','L-PC']].max(axis=1)
        hist['ATR']=hist['TR'].rolling(20).mean()
        return hist
    
    def OBV(self,hist):
        hist['OBV']=np.where(hist['Close'] > hist['Close'].shift(1), hist['Volume'],\
        np.where(hist['Close'] < hist['Close'].shift(1),-hist['Volume'],0)).cumsum()
        return hist
    
    def analyser(self, hist, mode = 'MACD', return_fig = False):
        if mode == 'MACD':
            hist = self.MACD(hist)
            columns = ['MACD', 'Signal', 'Histogram']
        elif mode == 'RSI':
            hist = self.RSI(hist)
            columns = ['RSI']
        elif mode == 'BollingerBands':
            hist = self.BollingerBands(hist)
            columns = ['Close', 'Upper', 'Lower', 'MA20']
        elif mode == 'ATR':
            hist = self.ATR(hist)
            columns = ['ATR']
        elif mode == 'OBV':
            hist = self.OBV(hist)
            columns = ['OBV']

        #st.stop()
        if return_fig:
            try:
                # fig with secondary y axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for c in columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist[c], mode='lines', name=c))
                # add hist as candlestick
                fig.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close']), secondary_y=True)
                    
                return fig
            except:
                pass
        else:
            return hist
        
class StocktonGym:
    def DEFAULT_CUSTOM_FUNCTION(self, hist, i):
        if random.randint(0, 200) < 75:
            action_string = 'buy'
        elif random.randint(0, 200) < 150:
            action_string = 'sell'
        else:
            action_string = 'hold'
        return action_string
    
    def __init__(self):
        # create a form to get the ticker and interval
        self.analysis_mode = st.sidebar.selectbox(label = 'Select Analysis Mode', options=['MACD', 'RSI', 'BollingerBands', 'ATR', 'OBV'])
        self.score_box = st.sidebar.empty()
        self.position_history_box = st.empty()
        c1,c2 = st.columns(2)
        self.plot_box_entire = st.empty()
        self.plot_box = c1.empty()
        self.plot_box_1  = c2.empty()
        self.plot_box_2  = st.empty()
        self.score_box_text = st.empty()

        self.form2 = st.sidebar.form(key='my_form2')
        c1,c2 = self.form2.columns(2)
        with self.form2:
            self.observer = st.checkbox('Observer Mode', value=False)
            self.initial_score = st.number_input(label='Capital', value=1000.00, step=1.00)
            self.start_simulation_button = c1.form_submit_button('Start Simulation', use_container_width=True)
            self.stop_simulation = c2.form_submit_button('Stop Simulation', use_container_width=True)
            self.pie_plot_box= st.empty()
            self.leverage = st.number_input(label='Leverage', value=1.00, step=1.00, min_value=1.00, max_value=100.00)

        self.form = st.sidebar.form(key='my_form')
        with self.form:
            self.submit_button = st.form_submit_button(label='Submit', use_container_width=True)
            self.ticker = st.selectbox(label = 'Select Ticker', options=['TSLA', 'eurusd=x', 'BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX', 'CMCSA', 'PEP', 'COST', 'TMUS', 'AVGO', 'QCOM', 'INTC', 'TXN', 'CHTR', 'SBUX', 'AMGN', 'AMD', 'GILD', 'BKNG', 'FISV', 'MDLZ', 'INTU', 'ISRG', 'ZM', 'ADP', 'MU', 'CSX', 'VRTX', 'ATVI', 'ILMN', 'REGN', 'ADI', 'BIIB', 'AMAT', 'NXPI', 'ADSK', 'MNST', 'LRCX', 'JD', 'EBAY', 'KHC', 'BIDU', 'WBA', 'MRNA', 'MELI', 'EXC', 'WDC', 'LULU', 'ROST', 'CTSH', 'EA', 'MAR', 'WDAY', 'ORLY', 'XEL', 'PAYX', 'DXCM', 'SNPS', 'NTES', 'CDNS', 'SGEN', 'VRSK', 'CTAS', 'CPRT', 'XLNX', 'FAST', 'MXIM', 'DLTR', 'SPLK', 'CERN', 'ANSS', 'SWKS', 'ASML', 'IDXX', 'CDW', 'CHKP', 'PCAR', 'VRSN', 'TCOM', 'ULTA', 'FOXA', 'FOX', 'SGMS'])
            self.interval = st.selectbox(label = 'Select Interval', options=['1m', '2m', '5m', '15m', '30m', '1h', '4h','1d', '5d', '1wk', '1mo', '3mo'], index=7)
            self.start_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
            self.end_date = st.date_input('End Date', value=pd.to_datetime('2022-12-31'))
            # submit button
        
        self.hist = self._get_data(self.interval)
        # buy and sell buttons
        self.position_history = pd.DataFrame(columns=['Date', 'Position_type', 'Open_price', 'Close_price', 'Profit', 'Difference_between_open_and_close'])

        self.custom_function_ = self.DEFAULT_CUSTOM_FUNCTION

    def custom_function(self, hist, i):
        return self.custom_function_(hist, i)
        
    def _handle_events(self):
        if self.stop_simulation:
            st.stop()
        if self.submit_button:
            self.hist = self._get_data(self.interval)
            # plot data 
            fig = go.Figure(data=[go.Candlestick(x=self.hist.index,
                        open=self.hist['Open'],
                        high=self.hist['High'],
                        low=self.hist['Low'],
                        close=self.hist['Close'])])
            self.plot_box_entire.plotly_chart(fig, use_container_width=True)
        if self.start_simulation_button:
            self.start_simulation()
    
    def _get_data(self, interval):
        interval = self.interval
        tick = yf.Ticker(self.ticker)
        hist = tick.history(interval=interval, start=self.start_date, end=self.end_date)
        return hist

    def operate_action(self, action_string, last_price, next_price_, position_history, hist):
        # Update position history and score 
        if action_string == 'buy': #tomorrow more than now
                position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1], 'Position_type': 'buy',
                                                                            'Open_price': last_price,
                                                                                'Close_price': next_price_, 
                                                                                'Profit': next_price_ - last_price,# * leverage,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
                profit = next_price_ - last_price
        
        elif action_string == 'sell': # now more than tomorrow
            position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1], 'Position_type': 'sell',
                                                                            'Open_price': last_price,
                                                                                'Close_price': next_price_, 
                                                                                'Profit': last_price - next_price_ ,# * leverage,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
            profit = last_price - next_price_
        elif action_string == 'hold':
            
            position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1], 'Position_type': 'hold',
                                                                            'Open_price': last_price,
                                                                                'Close_price': next_price_, 
                                                                                'Profit': 0,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
            profit = 0
        else:
            profit = 0
        return profit, position_history
      
    def start_simulation(self):
        # initialize the score
        score = self.initial_score
        scores = []
        hist_complete = self.hist
        # start the loop
        custom_expander = st.sidebar.expander(label='Custom Function Output')
        for i in range(hist_complete.shape[0]):
            i = i + 1
            hist = hist_complete.iloc[0:i]
            with custom_expander:
                action_string = self.custom_function(hist = hist, i = i)#, leverage=self.leverage)
            if i > 3:
                next_price = hist['Close'].iloc[-1]
                last_price = hist['Close'].iloc[-2]
                
                # operate the action
                profit, position_history = self.operate_action(action_string, last_price, next_price, self.position_history, hist)
                score = score + profit
                scores.append(score)

                if score > 0:
                    self.score_box.success(f'Capital: {score} £')
                else:
                    self.score_box.error(f'Capital: {score} £')

                self.position_history = position_history
                # only sell and buy 
                df_pos = self.position_history[self.position_history['Position_type'] != 'hold'] 
                # add index
                df_pos = df_pos.reset_index(drop=True)

                # reverse the order
                df_pos = df_pos.iloc[::-1]
                self.position_history_box.dataframe(df_pos, use_container_width=True)
            
            fig = go.Figure()
            # add the candlestick
            fig.add_trace(go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close']))
            # add annotations for buy and sell
            for index, row in self.position_history.iterrows():
                if row['Position_type'] == 'buy':
                    fig.add_annotation(x=row['Date'], y=row['Open_price'], text="B", showarrow=False, arrowhead=1, font=dict(color="green", size=14))
                elif row['Position_type'] == 'sell':
                    fig.add_annotation(x=row['Date'], y=row['Open_price'], text="S", showarrow=False, arrowhead=1, font=dict(color="red", size=14))
                elif row['Position_type'] == 'hold':
                    fig.add_annotation(x=row['Date'], y=row['Open_price'], text="H", showarrow=False, arrowhead=1, font=dict(color="blue", size=14))
            # add macd
            self.plot_box.plotly_chart(fig, use_container_width=True)
            analyser_fig = Analyser().analyser(hist, mode=self.analysis_mode, return_fig=True)
            self.plot_box_2.plotly_chart(analyser_fig, use_container_width=True)
            
            fig = go.Figure()
            # add the score
            fig.add_trace(go.Scatter(x=hist.index, y=scores, mode='lines', name='Profit'))
            # add hist as candlestick
          
            self.plot_box_1.plotly_chart(fig, use_container_width=True)
            # add the pie chart
            # filter the position history for buy and sell
            buy_ = self.position_history[self.position_history['Position_type'] == 'buy']
            sell_ = self.position_history[self.position_history['Position_type'] == 'sell']
            hold = self.position_history[self.position_history['Position_type'] == 'hold']

            number_of_buys = buy_.shape[0]
            number_of_sells = sell_.shape[0]
            number_of_holds = hold.shape[0]
            fig = go.Figure(data=[go.Pie(labels=['Buy', 'Sell', 'Hold'], values=[number_of_buys, number_of_sells, number_of_holds])])
            self.pie_plot_box.plotly_chart(fig, use_container_width=True)
            # wait for a second
            self.score_box_text.subheader(f'N iterations: {i}')

            
            if self.observer:
                time.sleep(0.5)

    def run(self):
        self._handle_events()

if __name__ == '__main__':
    gym = StocktonGym()
    
    def custom_function_(hist):
        '''
        Random function
        '''
        if random.randint(0, 200) < 75:
            action_string = 'buy'
        elif random.randint(0, 200) < 150:
            action_string = 'sell'
        else:
            action_string = 'hold'
        return action_string
    
    gym.custom_function_ = custom_function_
    
    gym.run()
