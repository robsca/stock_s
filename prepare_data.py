''' 
The idea is that we give the model the minimum amount of information and let it figure out the rest.
- take the data for the month of January

'''
import yfinance as yf
from utils import evaluate_support_resistance
from model_base import Linear_QNet, QTrainer
import random
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

ticker = st.selectbox(label = 'Select Ticker', options=['TSLA', 'eurusd=x', 'BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX', 'CMCSA', 'PEP', 'COST', 'TMUS', 'AVGO', 'QCOM', 'INTC', 'TXN', 'CHTR', 'SBUX', 'AMGN', 'AMD', 'GILD', 'BKNG', 'FISV', 'MDLZ', 'INTU', 'ISRG', 'ZM', 'ADP', 'MU', 'CSX', 'VRTX', 'ATVI', 'ILMN', 'REGN', 'ADI', 'BIIB', 'AMAT', 'NXPI', 'ADSK', 'MNST', 'LRCX', 'JD', 'EBAY', 'KHC', 'BIDU', 'WBA', 'MRNA', 'MELI', 'EXC', 'WDC', 'LULU', 'ROST', 'CTSH', 'EA', 'MAR', 'WDAY', 'ORLY', 'XEL', 'PAYX', 'DXCM', 'SNPS', 'NTES', 'CDNS', 'SGEN', 'VRSK', 'CTAS', 'CPRT', 'XLNX', 'FAST', 'MXIM', 'DLTR', 'SPLK', 'CERN', 'ANSS', 'SWKS', 'ASML', 'IDXX', 'CDW', 'CHKP', 'PCAR', 'VRSN', 'TCOM', 'ULTA', 'FOXA', 'FOX', 'SGMS'])
c1,c2 = st.sidebar.columns(2)
start_simulation = c1.button('Start Simulation')
stop_simulation = c2.button('Stop Simulation')
button_restart = st.sidebar.button('Restart Model')
score = st.sidebar.slider('Capital', min_value=0, max_value=100000, value=500)
leverage = st.sidebar.slider('Leverage', min_value=1, max_value=100, value=1)
if stop_simulation:
    st.stop()


if button_restart:
    try:
        # delete model and restart
        import os
        # remove folder 
        os.system('rm -rf model')
        # remove file
        os.system('rm model.pth')
    except:
        pass


if start_simulation:
    st.write('Simulation started')
    # a month of data
    interval = '1h'
    import pandas as pd
    eur_usd = yf.Ticker(ticker)
    # start date is a week before the end date
    # same for february
    hist_dec_to_feb = eur_usd.history(interval=interval, start='2021-12-01', end='2023-02-01')
    hist_feb_to_may = eur_usd.history(interval=interval, start='2023-02-01', end='2023-05-31')
    hist_may_to_today= eur_usd.history(interval=interval, start='2023-05-01', end='2023-07-21')
    hist_feb_to_dec = pd.concat([hist_dec_to_feb , hist_feb_to_may, hist_may_to_today])
    hist_feb = hist_feb_to_dec

    # the first state will be done with january and after that we are going to feed at each interval the next state
    from utils import evaluate_support_resistance_for_ML
    resistance, support, list_counts, current_price = evaluate_support_resistance(hist_dec_to_feb, verbose=False, sensibility=5)
    supports_and_resistances = list_counts
    print(supports_and_resistances)
    # get index of current price
    index_current_price = supports_and_resistances.index(current_price)
    print(index_current_price)
    print(len(supports_and_resistances))

    # now feed each day of february and calculate the resistance and support
    number_of_inputs = 52
    number_of_hidden_neurons = 256
    number_of_outputs = 3

    # try to load the model
    try:
        model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
        model.load()
    except:
        model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
    trainer = QTrainer(model, 0.001, 0.9)

    def get_state(i):
        # current price is the close of the previous day
        new_price = hist_feb.iloc[i]['Close']
        average = hist_feb.iloc[:i]['Close'].mean()
        average = [average,new_price]
        #print('This is the new price')
        #print(new_price)
        # merge jan and feb 
        hist_for_support_resistance = pd.concat([hist_dec_to_feb, hist_feb.iloc[0:i]])
        # keep last len(hist_jan)
        hist_for_support_resistance = hist_for_support_resistance
        resistance, support, list_counts, current_price = evaluate_support_resistance_for_ML(hist_for_support_resistance, verbose=False, sensibility=5)
        supports_and_resistances = list_counts
        #print('This is the list of supports and resistances')
        #print(supports_and_resistances)
        # get index of current price
        try:
            index_current_price = supports_and_resistances.index(new_price)
        except:
            # get the closest price
            index_current_price = min(supports_and_resistances, key=lambda x:abs(x-new_price))
            index_current_price = supports_and_resistances.index(index_current_price)
        list_of_zeros = [0] * len(supports_and_resistances)
        list_of_zeros[index_current_price] = 1
        #print('This is the list of zeros')
        print(list_of_zeros)
        state = list_of_zeros + average
        return state
    
    def get_reward(state, action, next_state):
        '''
        Options: tied to the current price and the next price
        let's suppose we have this:
        state =  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        action = 0 (buy)
        next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        reward = 1
        '''
        current_price = state[-1]
        next_price = next_state[-1]
        if action == 0: # buy
            # find index of 1 in state
            index_current_price = state.index(1)
            index_current_price_next_state = next_state.index(1)
            if index_current_price_next_state > index_current_price and next_price > current_price:
                return 1
            elif index_current_price_next_state < index_current_price and next_price < current_price:
                return -1
            else:
                return 0
        elif action == 1: # sell
            # find index of 1 in state
            index_current_price = state.index(1)
            index_current_price_next_state = next_state.index(1)
            if index_current_price_next_state < index_current_price and next_price < current_price:
                return 1
            elif index_current_price_next_state > index_current_price and next_price > current_price:
                return -1
            else:
                return 0
        elif action == 2: # hold
            return 0
        #

    def get_action(with_string=False):
        '''
        At the moment is a simple random choice between the 3 options.
        
        Goal
        ---
        Create a order buy, or sell
        '''
        classes = ['buy', 'sell', 'hold']
        indexes = [0,1,2]
        action = random.choice(indexes)
        if with_string:
            return action, classes[action]
        return action
            
    def get_next_state(i):
        try:
            state = get_state(i+1)
        except:
            state = get_state(i)
        return state

    def get_done(i, score):
        '''
        if finish row return true otherwise return false
        '''
        if i == len(hist_feb) - 1 or score < 0:
            done = [1]
            st.stop()
        else:
            return None
        return done

    import torch
    def get_action_from_state(state, with_string=False):
        # here we are using the model to get the action
        # tranform in tensor
        state = torch.tensor(state, dtype=torch.float)
        if with_string:
            return torch.argmax(model(state)).item(), get_action(with_string=True)
        return torch.argmax(model(state)).item()
        


    def training_loop(epochs):
        global score, leverage
        saving_model = st.empty()
        score_box = st.empty()
        date_box = st.empty()
        c1,c2 = st.columns(2)
        message = st.empty()
        after_message = st.empty()
        plot = c1.empty()
        plot_score = c2.empty()
        scores = []
        best_score = score
        position_history_box = st.empty()
        position_history = pd.DataFrame(columns=['Date', 'Position_type', 'Open_price', 'Close_price', 'Profit', 'Difference_between_open_and_close'])
        for i in range(epochs):
            i = i + 1
            hist = hist_feb.iloc[0:i]
            try:
                hist_with_next_price = hist_feb.iloc[0:i+1]
                last_price = hist.iloc[-1]['Close']
                next_price_ = hist_with_next_price.iloc[-1]['Close']
            except:
                last_price = hist.iloc[-1]['Close']
                next_price_ = hist.iloc[-1]['Close']
            # update position history
            state = get_state(i)
            action, action_string = get_action_from_state(state, with_string=True)
            next_state = get_next_state(i)
            reward = get_reward(state, action, next_state)

            # update position history
            if action_string[1] == 'buy':
                if reward == -1: # mistake means that we buy and the next price is lower than the current price
                    position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1], 'Position_type': 'buy',
                                                                                'Open_price': last_price,
                                                                                    'Close_price': next_price_, 
                                                                                    'Profit': (last_price - next_price_) * leverage,
                                                                                    'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
                    profit = last_price - next_price_
                elif reward == 1:
                    position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1],
                                                                                'Position_type': 'buy', 
                                                                                'Open_price': last_price, 
                                                                                'Close_price': next_price_, 
                                                                                'Profit': (next_price_ - last_price) * leverage,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
                    profit = next_price_ - last_price
                else:
                    profit = 0
            elif action_string[1] == 'sell':
                if reward == -1:
                    # add to position history multiplying
                    position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1], 
                                                                                'Position_type': 'sell', 
                                                                                'Open_price': last_price, 
                                                                                'Close_price': next_price_, 
                                                                                'Profit': (next_price_ - last_price) * leverage,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
                    profit = next_price_ - last_price
                elif reward == 1:
                    # multiply the profit by -1
                    position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1],
                                                                                'Position_type': 'sell', 
                                                                                'Open_price': last_price, 
                                                                                'Close_price': next_price_, 
                                                                                'Profit': (last_price - next_price_) * leverage,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
                    profit = last_price - next_price_
                else:
                    profit = 0
            elif action_string[1] == 'hold':
                # hold position
                position_history = pd.concat([position_history, pd.DataFrame({'Date': hist.index[-1],
                                                                                'Position_type': 'hold', 
                                                                                'Open_price': last_price, 
                                                                                'Close_price': next_price_, 
                                                                                'Profit': 0,
                                                                                'Difference_between_open_and_close': next_price_ - last_price}, index=[0])])
                profit = 0
            else:
                profit = 0

            # multiply profit by leverage
            profit = profit * leverage

            # reverse the dataframe
            position_history = position_history.iloc[::-1]
            position_history_box.dataframe(position_history)
            # update score
            score = score + profit

            done = get_done(i, score)
            trainer.train_step(state, action, reward, next_state, done)
            print('Action: ', action_string)
            print('Reward: ', reward)
            print('Next State: ', next_state)
            print('Done: ', done)
            print('Score: ', score)

            scores.append(score)
            if score < 0:
                break
            # update best score
            if best_score < score:
                best_score = score
                saving_model.success(f'Best Score: {best_score} - Saving model at epoch: {i}')
                model.save()

            print(i)
            # plot
            fig_score = go.Figure(data=go.Scatter(y=scores))
            plot_score.plotly_chart(fig_score, use_container_width=True)
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'])])
            plot.plotly_chart(fig, use_container_width=True)
            # update message
            message.text(f'Epoch: {i} - Score: {score}')
            score_box.subheader(f'Capital: {score},00 £')
            date_box.subheader(f'Date: {hist.index[-1]}')
            # save the model at the end of the training
            after_message.text(f'Action: {action_string} - Reward: {reward} - Next State: {next_state} - Done: {done} - Score: {score} - Best Score: {best_score}')
            if i == epochs - 1:
                model.save()
                saving_model.write(f'Best Score: {best_score} - Saving model at epoch: {i}')
                break
                

        print('Model Score: ', score, ' out of ', epochs)

    training_loop(len(hist_feb))