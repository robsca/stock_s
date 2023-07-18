
'''
author: Roberto Scalas 
date:   2023-07-17 10:34:58.351165
'''
from utils import *
from datab import Database_Transactions

# delete the database

db_transactions = Database_Transactions()
db_transactions.delete_table()
db_transactions.close()