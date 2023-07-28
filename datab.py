import sqlite3 
# create a database connection
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

class Database_Scripts:
    def __init__(self, table_name = '`stockton.projects`', table_name_for_json = 'stockton.projects'):
        self.table_name = table_name
        self.table_name_for_json = table_name_for_json
        credentials = service_account.Credentials.from_service_account_file(st.secrets["gcp_service_account"])
        self.client = bigquery.Client(credentials=credentials)

    # why this function isn't actually working when using the form?
    def insert(self, Project_name, Model, Agent, Utils):
        self.client.insert_rows_json(self.table_name_for_json, [{'Project_name': Project_name, 'Model': Model, 'Agent': Agent, 'Utils': Utils}])
        #self.client.query(query_in_different_format)
    
    def select(self):
        result = self.client.query(f"SELECT * FROM {self.table_name}")
        return result
    
    def delete(self):
        self.client.query(f"DELETE FROM {self.table_name}")

    def delete_table(self):
        self.client.query(f"DROP TABLE {self.table_name}")

    def get_from_project(self, project_name):
        project = self.client.query(f"SELECT * FROM {self.table_name} WHERE Project_name = '{project_name}'")
        return project
    
    def update_from_project(self, project_name, Model, Agent, Utils):
        self.client.query(f"UPDATE {self.table_name} SET Model = {Model}, Agent = {Agent}, Utils = {Utils} WHERE Project_name = {project_name}")
    
    def delete_from_project(self, project_name):
        self.client.query(f"DELETE FROM `stockton.projects` WHERE Project_name = '{project_name}'")

    # update agent from project
    def update_agent_from_project(self, project_name, Agent):
        self.client.query(f"UPDATE {self.table_name} SET Agent = {Agent} WHERE Project_name = '{project_name}'")

    # update model from project
    def update_model_from_project(self, project_name, Model):
        self.client.query(f"UPDATE {self.table_name} SET Model = {Model} WHERE Project_name = '{project_name}'")

    # update utils from project
    def update_utils_from_project(self, project_name, Utils):
        self.client.query(f"UPDATE {self.table_name} SET Utils = {Utils} WHERE Project_name = '{project_name}'")
    # get agent from project

    def get_agent_from_project(self, project_name):
        res = self.client.query(f"SELECT Agent FROM {self.table_name} WHERE Project_name = '{project_name}'")
        row_iterator = res.result()
        rows = list(row_iterator)
        return rows
    
    # get model from project
    def get_model_from_project(self, project_name):
        res = self.client.query(f"SELECT Model FROM {self.table_name} WHERE Project_name = '{project_name}'")
        row_iterator = res.result()
        rows = list(row_iterator)
        return rows
    
    
    # get utils from project
    def get_utils_from_project(self, project_name):
        res = self.client.query(f"SELECT Utils FROM {self.table_name} WHERE Project_name = '{project_name}'")
        row_iterator = res.result()
        rows = list(row_iterator)
        return rows

class Database_Questions:
    def __init__(self, table_name = '`stockton.conversations`', table_name_for_json = 'stockton.conversations'):
        self.table_name = table_name
        self.table_name_for_json = table_name_for_json
        credentials = service_account.Credentials.from_service_account_file(st.secrets["gcp_service_account"])
        self.client = bigquery.Client(credentials=credentials)

    def insert(self, conversation_id, question, answer, date):
        self.client.insert_rows_json(self.table_name_for_json, [{'conversation_id': conversation_id, 'question': question, 'answer': answer, 'date': date}])
    
    def select(self):
        result = self.client.query(f"SELECT * FROM {self.table_name}")
        row_iterator = result.result()
        rows = list(row_iterator)
        return rows
    
    def delete(self):
        self.client.query(f"DELETE FROM {self.table_name}")

    def delete_single(self, conversation_id):
        self.client.query(f"DELETE FROM {self.table_name} WHERE conversation_id = '{conversation_id}'")
        
                
    def delete_table(self):
        self.client.query(f"DROP TABLE {self.table_name}")

    def get_from_conversation_id(self, conversation_id):
        res = self.client.query(f"SELECT * FROM {self.table_name} WHERE conversation_id = '{conversation_id}'")
        row_iterator = res.result()
        rows = list(row_iterator)
        return rows
    
    
    def delete_single_conversation_form_id(self, conversation_id):
        # delete the conversation
        self.client.query(f"DELETE FROM `stockton.conversations` WHERE conversation_id = '{conversation_id}'")
        # now print the result
        res = self.client.query(f"SELECT * FROM {self.table_name}")
        row_iterator = res.result()
        rows = list(row_iterator)
        print(rows)
        print('deleted')
if __name__ == '__main__':
    # create a project
    db = Database_Scripts('Projects')
    db.insert('Test', 'Model', 'Agent', 'Utils')
    print(db.select())
    db.close()


