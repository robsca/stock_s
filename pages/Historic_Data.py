''' 
The idea is that we give the model the minimum amount of information and let it figure out the rest.
- take the data for the month of January

'''
from gym_stockton import StocktonGym
import random
from agent_stockton import AGENT
import streamlit as st

if __name__ == '__main__':
    gym = StocktonGym()
    agent = AGENT()

    if st.button('Restart'):
        agent.restart_model()
    gym.custom_function_ = agent.step_agent    
    gym.run()