
'''
author: Roberto Scalas 
date:   2023-07-17 10:34:58.351165
'''
from scripts.utils import *
from datab import Database_Transactions

# delete the database

# db_transactions = Database_Transactions()
# db_transactions.delete_table()
# db_transactions.close()

import streamlit as st
from streamlit_ace import st_ace

code = '''
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
options_for_ticker = [
    'eurusd=x',
    'gbpusd=x',
    'usdjpy=x',
    'usdchf=x',
    'TSLA',
    'AAPL',
    'MSFT',
    'AMZN',
    'GOOG',
    'VUSA.L',

]
import datetime
ticker = st.sidebar.selectbox('Select ticker', options_for_ticker)
interval = st.sidebar.selectbox('Select interval', ['1d', '1h', '30m', '15m', '5m', '1m'])
start_date = st.sidebar.date_input('Start date', value=datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input('End date', value=datetime.date(2021, 7, 1))
data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
'''
code = st_ace(value=code, language='python', theme='monokai', keybinding='vscode', font_size=12, tab_size=4, show_gutter=True, show_print_margin=True, wrap=True, auto_update=True, readonly=False, key=None)

# Store the code as a string
stored_code = code

# Later, execute the stored code to define the function
exec(stored_code)

# Now, you can call the dynamically defined function
result = function()
st.write(result)  # Output: ciao