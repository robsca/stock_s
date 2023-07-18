
import streamlit as st
st.set_page_config(layout="wide")
from utils import *
import datetime
import pandas as pd
from datab import Database_Transactions

class Application:
    def __init__(self):
        self.db_transactions = Database_Transactions()

    def elements_UI(self):
        st.title('Transactions History')
        expander_t = st.expander('Transactions History', expanded= True)
        table_transaction_inactive = expander_t.empty()
        expander_t_active = st.expander('Active Transactions', expanded= True)
        table_transaction = expander_t_active.empty()
        space_res = st.empty()

        return table_transaction_inactive, table_transaction, space_res

    async def main(self):
        table_transaction_inactive, table_transaction_active, result_expander = self.elements_UI()
        period = '1d'
        interval = '1m'

        # get current price
        self.transactions = self.db_transactions.select()
        clear_button = st.sidebar.button('Clear transactions')
        # add a button to clear the transactions
        if clear_button:
            self.db_transactions.delete()
            self.transactions = self.db_transactions.select()

        transaction_expander = st.sidebar.expander('Transactions', expanded= True)
        with transaction_expander:
            for transaction in self.transactions:
                # add a button to close the position
                if transaction[6] == 1:
                    st.write(transaction[6])
                    st.table(transaction)
                    if st.button('Close position', key=transaction):
                        ticker = transaction[5]
                        current_price = await get_current_price(ticker, period=period, interval=interval)
                        self.db_transactions.close_position(transaction[0], transaction[5], current_price, current_price * transaction[3], (current_price * transaction[3]) - transaction[4], ((current_price * transaction[3]) - transaction[4]) / transaction[4] * 100)
                        self.transactions = self.db_transactions.select()
                        break
        while True: 
            self.transactions = Database_Transactions().select()
            transactions_for_result = pd.DataFrame(self.transactions, columns=['Date', 'Type', 'Price', 'Quantity', 'Total', 'Ticker', 'Active', 'Close_price', 'Close_value', 'Result', 'Result_percent'])
            # get number of active transactions
            number_active_transactions = len(transactions_for_result.loc[transactions_for_result['Active'] == 1])
            if number_active_transactions > 0:
                transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_price'] = await get_current_price(transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Ticker'].values[0], period=period, interval=interval)
                transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_value'] = transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_price'] * transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Quantity']
                transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Result'] = (transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_price'] * transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Quantity']) - transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Total']
                transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Result_percent'] = ((transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_price'] * transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Quantity']) - transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Total']) / transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Total'] * 100

            active_transactions = transactions_for_result.loc[transactions_for_result['Active'] == 1]
            inactive_transactions = transactions_for_result.loc[transactions_for_result['Active'] == 0]
            if len(active_transactions) > 0:
                table_transaction_active.table(active_transactions)
            if len(inactive_transactions) > 0:
                table_transaction_inactive.table(inactive_transactions)
            result_expander.subheader(f'Total result: {transactions_for_result["Result"].sum()}')

if __name__ == "__main__":
    asyncio.run(Application().main())