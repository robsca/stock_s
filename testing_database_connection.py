from google.cloud import bigquery
from google.oauth2 import service_account



# Create a query job
query_job = client.query("""
    SELECT *
    FROM `stockton.projects`
""")

# Fetch the results

query_job_2 = client.query("""
INSERT INTO `stockton.projects` Project_name, Model, Agent, Utils) VALUES ('test_1', 'test_1', 'test_1', 'test_1')""")

results = query_job.result()

class Database_Scripts:
    def __init__(self, table_name = '`stockton.projects`'):
        self.table_name = table_name
        credentials = service_account.Credentials.from_service_account_file('credentials.json')
        self.client = bigquery.Client(credentials=credentials)

    def insert(self, Project_name, Model, Agent, Utils):
        self.client.query(f"INSERT INTO {self.table_name} VALUES ('{Project_name}', '{Model}', '{Agent}', '{Utils}')")

    def select(self):
        result = self.client.query(f"SELECT * FROM {self.table_name}")
        return result
    
    def delete(self):
        self.client.query(f"DELETE FROM {self.table_name}")

    def close(self):
        self.conn.close()

    def delete_table(self):
        self.client.query(f"DROP TABLE {self.table_name}")
        self.conn.commit()

    def get_from_project(self, project_name):
        self.client.query(f"SELECT * FROM {self.table_name} WHERE Project_name = {project_name}")
        return self.cur.fetchall()
    
    def update_from_project(self, project_name, Model, Agent, Utils):
        self.client.query(f"UPDATE {self.table_name} SET Model = {Model}, Agent = {Agent}, Utils = {Utils} WHERE Project_name = {project_name}")
    
    def delete_from_project(self, project_name):
        self.client.query(f"DELETE FROM `stockton.projects` WHERE Project_name = '{project_name}'")

    # update agent from project
    def update_agent_from_project(self, project_name, Agent):
        self.client.query(f"UPDATE {self.table_name} SET Agent = {Agent} WHERE Project_name = {project_name}")

    # update model from project
    def update_model_from_project(self, project_name, Model):
        self.client.query(f"UPDATE {self.table_name} SET Model = {Model} WHERE Project_name = {project_name}")

    # update utils from project
    def update_utils_from_project(self, project_name, Utils):
        self.client.query(f"UPDATE {self.table_name} SET Utils = {Utils} WHERE Project_name = {project_name}")
    
    # get agent from project
    def get_agent_from_project(self, project_name):
        res = self.client.query(f"SELECT Agent FROM {self.table_name} WHERE Project_name = {project_name}")
        return res
    
    # get model from project
    def get_model_from_project(self, project_name):
        res = self.client.query(f"SELECT Model FROM {self.table_name} WHERE Project_name = {project_name}")
        return res
    
    # get utils from project
    def get_utils_from_project(self, project_name):
        res = self.client.query(f"SELECT Utils FROM {self.table_name} WHERE Project_name = {project_name}")
        return res
    
# delete a single project from the database

sql = """
DELETE FROM `stockton.projects`
WHERE Project_name = 'test_1'
"""
