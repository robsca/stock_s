import sqlite3 

# create a database connection
class Database_Transactions:
    def __init__(self):
        self.conn = sqlite3.connect('transactions.db')
        self.cur = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute('''CREATE TABLE IF NOT EXISTS transactions
            (Date text, Type text, Price real, Quantity integer, Total real, Ticket text, Active integer, Close_price real, Close_value real, Result real, Result_percent real)''')
        self.conn.commit()

    def insert(self, Date, Type, Price, Quantity, Total, Ticker, Active=1, Close_price=0, Close_value=0, Result=0, Result_percent=0):
        self.cur.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (Date, Type, Price, Quantity, Total, Ticker, Active, Close_price, Close_value, Result, Result_percent))
        self.conn.commit()

    def select(self):
        self.cur.execute("SELECT * FROM transactions")
        return self.cur.fetchall()
    
    def delete(self):
        self.cur.execute("DELETE FROM transactions")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def delete_table(self):
        self.cur.execute("DROP TABLE transactions")
        self.conn.commit()

    def close_position(self, date, ticker, current_price, current_value, result, result_percent):
        # when closing a position set active to 0 and add the current price and value
        self.cur.execute("UPDATE transactions SET Active = 0, Close_price = ?, Close_value = ?, Result = ?, Result_percent = ? WHERE Date = ? AND Ticket = ?", (current_price, current_value, result, result_percent, date, ticker))        
        self.conn.commit()


# create a database to store the scripts

class Database_Scripts:
    def __init__(self, table_name):
        self.table_name = table_name
        self.conn = sqlite3.connect('projects.db')
        self.cur = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # each table contains a model, agent and utils script
        self.cur.execute(f'''CREATE TABLE IF NOT EXISTS {self.table_name}   
            (Model text, Agent text, Utils text)''')
        self.conn.commit()

    def insert(self, Model, Agent, Utils):
        self.cur.execute(f"INSERT INTO {self.table_name} VALUES (?, ?, ?)", (Model, Agent, Utils))
        self.conn.commit()

    def select(self):
        self.cur.execute(f"SELECT * FROM {self.table_name}")
        return self.cur.fetchall()
    
    def delete(self):
        self.cur.execute(f"DELETE FROM {self.table_name}")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def delete_table(self):
        self.cur.execute(f"DROP TABLE {self.table_name}")
        self.conn.commit()

    def update(self, Model, Agent, Utils):
        self.cur.execute(f"UPDATE {self.table_name} SET Model = ?, Agent = ?, Utils = ?", (Model, Agent, Utils))
        self.conn.commit()

    def select_model(self):
        self.cur.execute(f"SELECT Model FROM {self.table_name}")
        return self.cur.fetchall()
    
    def select_agent(self):
        self.cur.execute(f"SELECT Agent FROM {self.table_name}")
        return self.cur.fetchall()
    
    def select_utils(self):
        self.cur.execute(f"SELECT Utils FROM {self.table_name}")
        return self.cur.fetchall()
    

if __name__ ==  '__main__':
    # create the database
    db = Database_Scripts('projects')
    # insert the defaults scripts into the database
    from pages.Editor import get_code
    def insert_default_scripts():
        # insert the default scripts into the database
        db.insert(get_code('scripts/script_default_model.py'), get_code('scripts/script_default_agent.py'), get_code('scripts/script_default_utils.py'))
        db.close()

    #insert_default_scripts()

    # check if the database is empty
    print(db.select())

