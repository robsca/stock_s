''' 
The idea is that we give the model the minimum amount of information and let it figure out the rest.
- take the data for the month of January

'''
from Gym import StocktonGym
import random
from scripts.agent import AGENT
import streamlit as st
from io import StringIO
from contextlib import redirect_stdout
from streamlit_ace import st_ace

if __name__ == '__main__':
    from pages.Editor import code_area
    from datab import Database_Scripts

    db = Database_Scripts('Projects')
    scripts = db.select()
    scripts = [script[0] for script in scripts]

    form_project = st.sidebar.form(key='form_project', clear_on_submit=False)
    project_name = form_project.selectbox('Select Project', scripts, key='project_name')
    request_button = form_project.form_submit_button(label='Load Model', use_container_width=True)
    st.write(project_name)

    if request_button:
        # write the code to the file
        agent = db.get_agent_from_project(project_name)[0][0]
        model = db.get_model_from_project(project_name)[0][0]
        utils = db.get_utils_from_project(project_name)[0][0]

        with open('scripts/agent.py', 'w') as f:
            f.write(agent)
        with open('scripts/model.py', 'w') as f:
            f.write(model)
        with open('scripts/utils.py', 'w') as f:
            f.write(utils)
        st.experimental_rerun()
        
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
  