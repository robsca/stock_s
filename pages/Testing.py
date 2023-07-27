
'''
author: Roberto Scalas 
date:   2023-07-17 10:34:58.351165
'''
from scripts.utils import *
import openai
import streamlit as st
st.set_page_config(layout="wide")

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
def function():
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    # add plotly graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],

                    close=data['Close'], name = 'market data'))
    fig.update_layout(
        title='{} Stock Chart'.format(ticker),
        yaxis_title='Stock Price (USD per Shares)')
    st.plotly_chart(fig, use_container_width=True)
    return data


'''
code = st_ace(value=code, language='python', theme='monokai', keybinding='vscode', font_size=12, tab_size=4, show_gutter=True, show_print_margin=True, wrap=True, auto_update=True, readonly=False, key=None)


def chat_with_gpt(prompt):
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    answer = chat_completion.choices[0].message['content']
    with st.expander('Answer'):
        st.markdown(answer)
    return answer

# create a chat with gpt
import datetime
from datab import Database_Questions
db = Database_Questions('Chat')
# get the conversations
conversations = db.select()
if len(conversations) > 0:
    # get unique conversations ids 
    conversations_ids = list(set([conversation[0] for conversation in conversations]))
    with st.sidebar.form(key='my_form_for_conversation'):
        conversation_id = st.selectbox('Select conversation', conversations_ids)
        submit_button = st.form_submit_button(label='Submit', help=None, on_click=None, args=None, kwargs=None)
        new_conversation = st.form_submit_button(label='New Chat', help=None, on_click=None, args=None, kwargs=None)
    if submit_button:
        # get all the questions and answers from the conversation
        questions_answers = db.get_from_conversation_id(conversation_id)
        with st.expander('Conversation'):
            # each row contains a question and an answer
            if len(questions_answers) > 0:
                for row in questions_answers:
                    # use the chat module
                    with st.chat_message('User'):
                        st.write(row[1])
                    with st.chat_message('GPT'):
                        st.write(row[2])
            else:
                st.write('No questions and answers for this conversation')
        # new conversation button
    if new_conversation:
        conversation_id = datetime.datetime.now()
        # reload the page
        # save a empty conversation
        db.insert(conversation_id, '', '', datetime.datetime.now())
        st.experimental_rerun()

    # create a delete button
    if st.sidebar.button(f'Delete conversation: {conversation_id}', use_container_width=True):
        db.delete_single(conversation_id)
        # reload the page
        st.experimental_rerun()
else:
    conversation_id = 'First conversation'


# create a new question form
with st.sidebar.form(key='my_form'):
    openai.api_key = st.sidebar.text_input('OpenAI API Key', value='', max_chars=None, key=None, type='password', help=None)
    question = st.text_area('Question', value='', max_chars=None, key=None)
    submit_button = st.form_submit_button(label='Submit', help=None, on_click=None, args=None, kwargs=None)

if submit_button:
    # get the answer and save it
    answer = chat_with_gpt(question)
    # get conversation id
    db.insert(conversation_id, question, answer, datetime.datetime.now())


# Store the code as a string
stored_code = code

# Later, execute the stored code to define the function
exec(stored_code)

# Now, you can call the dynamically defined function
result = function()
st.write(result)  # Output: ciao
