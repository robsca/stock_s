import plotly.graph_objects as go
import asyncio
import yfinance as yf
import numpy as np


def evaluate_support_resistance_for_ML(hist, verbose = True, sensibility = 5):
    '''
    This function evaluates the support and resistance of a stock
    - get min and max
    - create 10 containers between min and max
    - count how many times the price is in the container
    - the container with the most counts is the support or the resistance
    is a resistance if the price is going down
    is a support if the price is going up
    ''' 
    # Get the max and the min
    max_price = hist['High'].max()
    min_price = hist['Low'].min()
    # Create the bins

    bins = np.linspace(min_price, max_price, 50)
    # from the bins create the ranges 
    ranges = {
        f'{(bins[i]+bins[i+1])/2}': [bins[i], bins[i+1]] for i in range(len(bins)-1)
    }
    #print(ranges)


    counts = {
        f'{(bins[i]+bins[i+1])/2}': len(hist[(hist['High'] >= bins[i]) & (hist['Low'] <= bins[i+1])]) for i in range(len(bins)-1) 
    }

    #print(counts)
    # keep only the strongest third dynamically
    how_many_to_keep = int(len(counts))# / sensibility)
    # sort the dictionary by value
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
    # keep only the first third
    counts = {k: v for k, v in list(counts.items())[0:how_many_to_keep]}


    # get the list of the ranges
    list_counts = list(counts.keys())
    current_price = hist['Close'].iloc[-1]
    # add it to the list
    list_counts.append(current_price)
    # transform the list in a list of floats
    list_counts = [float(i) for i in list_counts]
    # sort the list
    list_counts = sorted(list_counts)

    # get the index of the current price
    index_current_price = list_counts.index(current_price)
    # get the index of the next price
    index_next_price = index_current_price + 1 if index_current_price < len(list_counts) - 1 else index_current_price
    # get the index of the previous price
    index_previous_price = index_current_price - 1 if index_current_price > 0 else index_current_price

    # get the next price
    resistance = list_counts[index_next_price]
    # list_resistances will be the list of the resistances
    list_resistances = list_counts[index_next_price:]
    list_supports = list_counts[:index_previous_price]
    # get the previous price
    support = list_counts[index_previous_price]
    if verbose:
        print('')
        print('This is the list of the ranges')
        print(list_counts)
        print('')
        print(f'The current price is {current_price}')
        print(f'The resistance is {resistance}')
        print(f'The support is {support}')
        print('')

    return resistance, support, list_counts, current_price, list_resistances, list_supports