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
    from datab import Database_Scripts

    agent = AGENT()
    if agent:
        st.sidebar.success('AGENT: {} - Loaded'.format(agent.model_name))
    if st.sidebar.button(f'Restart Weights and Memory: {agent.model_name}', use_container_width=True):
        agent.restart_model()
  
    db = Database_Scripts('Projects')
    scripts = db.select()
    scripts = [script[0] for script in scripts]
    with st.form(key='my_form_for_project'):
        project_name = st.selectbox('Select Project', scripts, key='project_name')
        request_button = st.form_submit_button(label='Load Model', use_container_width=True)
        if request_button:
            # write the code to the file
            agent_code = db.get_agent_from_project(project_name)[0][0]
            model_code = db.get_model_from_project(project_name)[0][0]
            utils_code = db.get_utils_from_project(project_name)[0][0]

            #st.write(agent_code)
            #st.write(model_code)
            #st.write(utils_code)

            # delete the files
            with open('scripts/agent.py', 'w') as f:
                f.write(agent_code)
            with open('scripts/model.py', 'w') as f:
                f.write(model_code)
            with open('scripts/utils.py', 'w') as f:
                f.write(utils_code)
                
    agent_code = db.get_agent_from_project(project_name)[0][0]
    model_code = db.get_model_from_project(project_name)[0][0]
    utils_code = db.get_utils_from_project(project_name)[0][0]

    gym = StocktonGym()
    # get the python script
    whole_script = agent_code + model_code + utils_code
    exec(whole_script)
    
    
    def custom_function(hist, i)
        with StringIO() as buf, redirect_stdout(buf):
            function(hist, i)
            return function(hist,i)

    #st.write(custom_function())
    gym.custom_function_ = custom_function
    gym.run()
    
