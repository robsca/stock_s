import sqlite3 
# create a database connection
class Database_Questions:
    def __init__(self, table_name):
        self.table_name = table_name
        self.conn = sqlite3.connect(table_name + '.db')
        self.cur = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # each table contains a model, agent and utils script
        self.cur.execute(f'''CREATE TABLE IF NOT EXISTS {self.table_name}
            (conversation_id text, question text, answer text, date text)''')
        self.conn.commit()

    def insert(self, conversation_id, question, answer, date):
        self.cur.execute(f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?)", (conversation_id, question, answer, date))
        self.conn.commit()

    def select(self):
        self.cur.execute(f"SELECT * FROM {self.table_name}")
        # drop if both question and answer are empty
        data = self.cur.fetchall()
        data = [row for row in data if row[1] != '' or row[2] != '']
        return data
    
    def delete(self):
        self.cur.execute(f"DELETE FROM {self.table_name}")
        self.conn.commit()

    def delete_single(self, conversation_id):
        self.cur.execute(f"DELETE FROM {self.table_name} WHERE conversation_id = ?", (conversation_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()

    def delete_table(self):
        self.cur.execute(f"DROP TABLE {self.table_name}")
        self.conn.commit()

    def get_from_conversation_id(self, conversation_id):
        self.cur.execute(f"SELECT * FROM {self.table_name} WHERE conversation_id = ?", (conversation_id,))
        return self.cur.fetchall()
    

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