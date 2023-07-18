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

