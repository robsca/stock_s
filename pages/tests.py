
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

code = st.text_area('Code', value='''def function():return "ciao"''', height=200)

# Store the code as a string
stored_code = code

# Later, execute the stored code to define the function
exec(stored_code)

# Now, you can call the dynamically defined function
result = function()
st.write(result)  # Output: ciao