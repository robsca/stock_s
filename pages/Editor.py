# get code from script_default_agent.py
from streamlit_ace import st_ace
import streamlit as st
st.set_page_config(layout="wide")

def get_code(path='scripts/script_default_agent.py'):
    python_script_path = path
    # read as string
    with open(python_script_path, 'r') as f:
        value = f.read()
        return value
c1,c2,c3 = st.columns(3)   
radio = c1.radio('Select a script', ['Agent', 'Model', 'Utils'], horizontal=True)

save = c2.button('Save', use_container_width=True)
restore_base = c3.button('Restore Default', use_container_width=True)
if radio == 'Agent':
    value = get_code('scripts/agent.py')
elif radio == 'Model':
    value = get_code('scripts/model.py')
elif radio == 'Utils':
    value = get_code('scripts/utils.py')


code_area = st_ace(value=value, language='python', theme='monokai', keybinding='vscode', font_size=14, tab_size=4, show_gutter=True, show_print_margin=True, wrap=True, auto_update=True, readonly=False, key=None)

if save and radio == 'Agent':
    with open('scripts/agent.py', 'w') as f:
        f.write(code_area)
    st.balloons()
    st.experimental_rerun()
elif save and radio == 'Model':
    with open('scripts/model.py', 'w') as f:
        f.write(code_area)
    st.balloons()
    st.experimental_rerun()
elif save and radio == 'Utils':
    with open('scripts/utils.py', 'w') as f:
        f.write(code_area)
    st.balloons()
    st.experimental_rerun()

if restore_base:
    # write the default script into the script_default_agent.py
    with open('scripts/agent.py', 'w') as f:
        f.write(get_code('scripts/script_default_agent.py'))

    # write the default script into the script_default_model.py
    with open('scripts/model.py', 'w') as f:
        f.write(get_code('scripts/script_default_model.py'))

    # write the default script into the script_default_utils.py
    with open('scripts/utils.py', 'w') as f:
        f.write(get_code('scripts/script_default_utils.py'))
    st.balloons()
    st.experimental_rerun()

# st.write(code_area)
