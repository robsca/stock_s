import pandas as pd
import yfinance as yf
from scripts.utils import *
from scripts.model import Linear_QNet, QTrainer
import random
import streamlit as st
import plotly.graph_objects as go
import torch
import os
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class AGENT:
    def __init__(self):
        self.model_name = 'Strategy_1_Agent'
        
    def moving_average_crossing_strategy(self,hist, i):
        short_window = 25
        long_window = 75
        data = hist[:i]
        signals = data
        signals['signal'] = 0.0
        signals['Close'] = data['Close']
        signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
        signals['diff'] = signals['short_mavg'] - signals['long_mavg']
        standard_deviation = signals['diff'].std()
        st.write(standard_deviation)
        signals['positions'] = ['sell' if x > standard_deviation else 'buy' if x < -standard_deviation else 'hold' for x in signals['diff']]
        last_signal = signals.iloc[-1]['positions']
        return last_signal


def function(hist,i):
    MAVerick = AGENT()
    signal = MAVerick.moving_average_crossing_strategy(hist,i)
    return signal
    
