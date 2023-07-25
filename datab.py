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
            (Project_name text, Model text, Agent text, Utils text)''')
        self.conn.commit()

    def insert(self, Project_name, Model, Agent, Utils):
        self.cur.execute(f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?)", (Project_name, Model, Agent, Utils))
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

    def get_from_project(self, project_name):
        self.cur.execute(f"SELECT * FROM {self.table_name} WHERE Project_name = ?", (project_name,))
        return self.cur.fetchall()
    
    def update_from_project(self, project_name, Model, Agent, Utils):
        self.cur.execute(f"UPDATE {self.table_name} SET Model = ?, Agent = ?, Utils = ? WHERE Project_name = ?", (Model, Agent, Utils, project_name))
        self.conn.commit()
    
    def delete_from_project(self, project_name):
        self.cur.execute(f"DELETE FROM {self.table_name} WHERE Project_name = ?", (project_name,))
        self.conn.commit()

    # update agent from project
    def update_agent_from_project(self, project_name, Agent):
        self.cur.execute(f"UPDATE {self.table_name} SET Agent = ? WHERE Project_name = ?", (Agent, project_name))
        self.conn.commit()

    # update model from project
    def update_model_from_project(self, project_name, Model):
        self.cur.execute(f"UPDATE {self.table_name} SET Model = ? WHERE Project_name = ?", (Model, project_name))
        self.conn.commit()

    # update utils from project
    def update_utils_from_project(self, project_name, Utils):
        self.cur.execute(f"UPDATE {self.table_name} SET Utils = ? WHERE Project_name = ?", (Utils, project_name))
        self.conn.commit()
    
    # get agent from project
    def get_agent_from_project(self, project_name):
        self.cur.execute(f"SELECT Agent FROM {self.table_name} WHERE Project_name = ?", (project_name,))
        return self.cur.fetchall()
    
    # get model from project
    def get_model_from_project(self, project_name):
        self.cur.execute(f"SELECT Model FROM {self.table_name} WHERE Project_name = ?", (project_name,))
        return self.cur.fetchall()
    
    # get utils from project
    def get_utils_from_project(self, project_name):
        self.cur.execute(f"SELECT Utils FROM {self.table_name} WHERE Project_name = ?", (project_name,))
        return self.cur.fetchall()
    

if __name__ == '__main__':
    # create a project
    db = Database_Scripts('Projects')
    db.insert('Test', 'Model', 'Agent', 'Utils')
    print(db.select())
    db.close()