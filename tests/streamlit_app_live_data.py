
'''
author: Roberto Scalas 
date:   2023-07-17 10:34:58.350766
'''

import streamlit as st
st.set_page_config(layout="wide")
from scripts.utils import *
import datetime
import pandas as pd
from datab import Database_Transactions

class Application:
    def __init__(self):
        self.db_transactions = Database_Transactions()
        self.transactions = None

    def elements_UI(self):
        title_box = st.empty()
        expander_t = st.sidebar.expander('Transactions History', expanded= True)
        table_transaction = expander_t.empty()
        space_res = st.empty()
        expander_open_positions = st.expander('Open Positions', expanded= True)
        table_open_positions = expander_open_positions.empty()
        suggestion_box = st.empty()

        hist_last_5 = st.empty()
        hist_cont = st.empty()
        c1,c2,c3 = st.columns(3)
        resistance_box = c1.empty()
        current_price_box = c2.empty()
        support_box = c3.empty()
        resistance_distance = c1.empty()
        support_distance = c3.empty()

        return table_transaction, hist_last_5, hist_cont, resistance_box, current_price_box, support_box, resistance_distance, support_distance, space_res, title_box, table_open_positions, suggestion_box

    async def main(self):
        table_transaction, hist_last_5, hist_cont, resistance_box, current_price_box, support_box, resistance_distance, support_distance, result_expander, title_box, table_open_positions, suggestion_box = self.elements_UI()
        c1,c2, c3 = st.sidebar.columns(3)
        ticker = c1.text_input('Ticker', 'VUSA.L')
        # add ticker to the title
        title_box.title(ticker)
        period = c2.selectbox('Period', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
        interval_def = dictionary_period_interval[period]
        interval = c3.selectbox('Interval', ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'], index = list(dictionary_period_interval.values()).index(interval_def))
        time_to_sleep = 60
        
        # sort the transactions to have only the ticker selected
        self.transactions = self.db_transactions.select()
        self.transactions = [transaction for transaction in self.transactions if transaction[5] == ticker]
        # get current price
        current_price = await get_current_price(ticker, period=period, interval=interval)

        # add a form to buy/sell
        my_form = st.sidebar.form(key='my_form')
        col1, col2, col3 = my_form.columns(3)
        type_transaction = col1.radio('Buy/Sell', ['Buy', 'Sell'])
        price = col2.number_input('Price', min_value=0.0, max_value=100000.0, value=current_price)
        quantity = col3.number_input('Quantity', min_value=0, max_value=100000, value=1)
        c1,c2 = my_form.columns(2)
        submit_button = c1.form_submit_button(label='Submit')
        clear_button = c2.form_submit_button(label='Clear')

        if submit_button:
            if type_transaction == 'Buy':
                # get price
                price = await get_current_price(ticker, period=period, interval=interval)
                total = price * quantity
                # add transacrion to the database
                self.db_transactions.insert(Date=datetime.datetime.now(), Type=type_transaction, Price=price, Quantity=quantity, Total=total, Ticker=ticker, Active=1)
            else:
                total = price * quantity
                # add transacrion to the database
                self.db_transactions.insert(Date=datetime.datetime.now(), Type=type_transaction, Price=price, Quantity=quantity, Total=total, Ticker=ticker, Active=1)
        
        self.transactions = self.db_transactions.select()
        # filter the transactions to have only the ticker selected
        self.transactions = [transaction for transaction in self.transactions if transaction[5] == ticker]


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
                        current_price = await get_current_price(ticker, period=period, interval=interval)
                        self.db_transactions.close_position(transaction[0], transaction[5], current_price, current_price * transaction[3], (current_price * transaction[3]) - transaction[4], ((current_price * transaction[3]) - transaction[4]) / transaction[4] * 100)
                        self.transactions = self.db_transactions.select()
                        break

                    
        list_count_empty = st.empty()
        clear_button_all = st.sidebar.button('Clear All Positions')
        hist = await get_stock_price(ticker, period, interval, time_to_sleep = 'dead')
        while True: 
            if clear_button_all:
                current_price = await get_current_price(ticker, period=period, interval=interval)
                for transaction in self.transactions:
                    if transaction[6] == 1:
                        self.db_transactions.close_position(transaction[0], transaction[5], current_price, current_price * transaction[3], (current_price * transaction[3]) - transaction[4], ((current_price * transaction[3]) - transaction[4]) / transaction[4] * 100)
                self.transactions = self.db_transactions.select()
                # refresh the page
                st.experimental_rerun()
            self.transactions = Database_Transactions().select()
            evaluation = evaluate_support_resistance(hist, verbose = False, sensibility = 3)
            resistance, support, list_counts, current_price = evaluation

            # add a label to each range and store in a dictionary 
            
            dict_range = {}
            for i, ran in enumerate(list_counts):
                if ran == current_price:
                    dict_range[f'Current price'] = list_counts[i]
                elif ran > current_price:
                    dict_range[f'Resistance {i}'] = list_counts[i]
                elif ran < current_price:
                    dict_range[f'Support {i}'] = list_counts[i]
            # sort by value
            dict_range = {k: v for k, v in sorted(dict_range.items(), key=lambda item: item[1])}
            # from big to small
            dict_range = dict(reversed(list(dict_range.items())))


            list_count_empty.write(dict_range)

            fig_last_5 = plot_candlestick(hist.tail(25), with_slider=False, support = support, resistance = resistance)
            hist_last_5.plotly_chart(fig_last_5, use_container_width=True)
            fig = plot_support_resistance(hist)
            hist_cont.plotly_chart(fig, use_container_width=True)

            resistance_box.success(f'Resistance: {resistance}')
            current_price_box.info(f'Current price: {current_price}')
            support_box.warning(f'Support: {support}')
            resistance_distance.success(f'Resistance distance: {(resistance - current_price) / current_price * 100}%')
            support_distance.warning(f'Support distance: {(current_price - support) / current_price * 100}%')

            transactions_for_result = self.transactions
            # as dataframe 
            transactions_for_result = pd.DataFrame(transactions_for_result, columns=['Date', 'Type', 'Price', 'Quantity', 'Total', 'Ticker', 'Active', 'Close_price', 'Close_value', 'Result', 'Result_percent'])
            # add the current price for the active transactions
            # filter the ticker
            transactions_for_result = transactions_for_result.loc[transactions_for_result['Ticker'] == ticker]
            transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_price'] = current_price
            transactions_for_result.loc[transactions_for_result['Active'] == 1, 'Close_value'] = current_price * transactions_for_result['Quantity']
            # result
            # create the result column using the close value
            transactions_for_result['Result'] = transactions_for_result.apply(lambda row: row['Close_value'] - row['Total'] if row['Type'] == 'Buy' else row['Total'] - row['Close_value'], axis=1)
            transactions_for_result['Result_percent'] = transactions_for_result.apply(lambda row: row['Result'] / row['Total'] * 100, axis=1)

            table_transaction.table(transactions_for_result)
            # get number of active transactions
            number_active_transactions = len(transactions_for_result.loc[transactions_for_result['Active'] == 1])
            if number_active_transactions > 0:
                table_open_positions.table(transactions_for_result.loc[transactions_for_result['Active'] == 1])
            else:
                table_open_positions.empty()
                
            result = transactions_for_result['Result'].sum()
            if result > 0:
                result_expander.success(f'Total result: {result}')
            elif result < 0:
                result_expander.error(f'Total result: {result}')

            number_resistance = len([i for i in list_counts if i > current_price])
            number_support = len([i for i in list_counts if i < current_price])
            if number_resistance > number_support:
                suggestion_box.success(f'Buy')
            elif number_resistance < number_support:
                suggestion_box.error(f'Sell')
            else:
                suggestion_box.warning(f'Wait')
            # get the index of the current price inside the list counts
            hist = await get_stock_price(ticker, period, interval, time_to_sleep = time_to_sleep)

if __name__ == "__main__":
    asyncio.run(Application().main())