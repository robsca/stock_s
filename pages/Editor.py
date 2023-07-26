# get code from script_default_agent.py
from streamlit_ace import st_ace
import streamlit as st
import tempfile

st.set_page_config(layout="wide")
from datab import Database_Scripts
# get all the scripts from the database

db = Database_Scripts('Projects')
scripts = db.select()
scripts = [script[0] for script in scripts]
if len(scripts) == 0:
    #
    st.info('No projects found. Create a new project.')
    st.stop()
    
project_name = st.sidebar.selectbox('Select a project', scripts)

def get_code(path='scripts/script_default_agent.py'):
    python_script_path = path
    # read as string
    with open(python_script_path, 'r') as f:
        value = f.read()
        return value

def get_code_from_db(project_name):
    try:
        # now get the code from the database
        db = Database_Scripts('Projects')
        scripts = db.get_from_project(project_name)
        scripts = scripts[0]
        return scripts
    except:
        st.stop()
    
c1,c2,c3 = st.columns(3)   
radio = c1.radio('Select a script', ['Agent', 'Model', 'Utils'], horizontal=True)

# Create a new project
my_form3 = st.sidebar.form(key='form3', clear_on_submit=True)
x1,x2 = my_form3.columns(2)
submit_button = x2.form_submit_button(label='Create new Project', use_container_width=True)
project_name_new = x1.text_input(label='Project Name', value='', autocomplete='on', key=None, help=None)
unique_project_name = True if project_name_new not in scripts else False
if submit_button and unique_project_name:
    db.insert(project_name_new, get_code('scripts/script_default_model.py'), get_code('scripts/script_default_agent.py'), get_code('scripts/script_default_utils.py'))
    st.experimental_rerun()

#
save = st.sidebar.button('Save', use_container_width=True)
restore_base = c3.button('Restore Default', use_container_width=True)

agent_code = db.get_agent_from_project(project_name)[0][0]
model_code = db.get_model_from_project(project_name)[0][0]
utils_code = db.get_utils_from_project(project_name)[0][0]

if radio == 'Agent':
    value = agent_code
elif radio == 'Model':
    value = model_code
elif radio == 'Utils':
    value = utils_code

code_area = st_ace(value=value, language='python', theme='monokai', keybinding='vscode', font_size=14, tab_size=4, show_gutter=True, show_print_margin=True, wrap=True, auto_update=True, readonly=False, key=None)

if save and radio == 'Agent':
    db.update_agent_from_project(project_name=project_name, Agent=code_area)
    agent_code = db.get_agent_from_project(project_name)[0][0]
     
elif save and radio == 'Model':
    db.update_model_from_project(project_name=project_name, Model=code_area)
    model_code = db.get_model_from_project(project_name)[0][0]
    # update the model from project

elif save and radio == 'Utils':
    db.update_utils_from_project(project_name=project_name, Utils=code_area)
    utils_code = db.get_utils_from_project(project_name)[0][0]

if restore_base:
    db.update_from_project(project_name=project_name, Model=get_code('scripts/script_default_model.py'), Agent=get_code('scripts/script_default_agent.py'), Utils=get_code('scripts/script_default_utils.py'))
    st.balloons()
    st.experimental_rerun()

delete_button = st.sidebar.button('Delete Project', use_container_width=True)
if delete_button:
    db.delete_from_project(project_name=project_name)
    st.experimental_rerun()

if st.button('Write'):
    with open('scripts/agent.py', 'w') as f:
        f.write(agent_code)
    with open('scripts/model.py', 'w') as f:
        f.write(model_code)
    with open('scripts/utils.py', 'w') as f:
        f.write(utils_code)
db.close()



# st.write(code_area)
