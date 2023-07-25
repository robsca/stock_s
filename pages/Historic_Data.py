''' 
The idea is that we give the model the minimum amount of information and let it figure out the rest.
- take the data for the month of January

'''
from gym_stockton import StocktonGym
import random
from scripts.agent import AGENT
import streamlit as st
from io import StringIO
from contextlib import redirect_stdout
from streamlit_ace import st_ace

if __name__ == '__main__':
    from pages.Editor import code_area
    gym = StocktonGym()
    # get the python script
    exec(code_area)
    
    def custom_function(hist):
        with StringIO() as buf, redirect_stdout(buf):
            function(hist)
            return function(hist)
        
    #st.write(custom_function())
    gym.custom_function_ = custom_function
    gym.run()
    

    agent = AGENT()
    if st.sidebar.button(f'Restart Model: {agent.model_name}'):
        agent.restart_model()
  