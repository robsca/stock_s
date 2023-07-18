import plotly.graph_objects as go
import asyncio
import yfinance as yf
import numpy as np

async def get_stock_price(ticker = 'VUSA.L', period = '5d', interval = '1d', start = None, end = None, actions = True, auto_adjust = True, back_adjust = False, time_to_sleep = 5):
    '''
    This function returns the historical data of a stock
    '''
    tesla = yf.Ticker(ticker)
    hist = tesla.history(period=period, interval=interval, start=start, end=end, actions=actions, auto_adjust=auto_adjust, back_adjust=back_adjust)
    if time_to_sleep != 'dead':
        await asyncio.sleep(time_to_sleep)
        return hist
    else:
        return hist
    
async def get_current_price(ticker = 'VUSA.L', period = '5d', interval = '1d', start = None, end = None, actions = True, auto_adjust = True, back_adjust = False, time_to_sleep = 5):
    '''
    This function returns the historical data of a stock
    '''
    tesla = yf.Ticker(ticker)
    hist = tesla.history(period=period, interval=interval, start=start, end=end, actions=actions, auto_adjust=auto_adjust, back_adjust=back_adjust)
    if time_to_sleep != 'dead':
        await asyncio.sleep(time_to_sleep)
        return hist['Close'].iloc[-1]
    else:
        return hist['Close'].iloc[-1]
     
def get_stock_price_hist(ticker = 'VUSA.L', period = '5d', interval = '1d', start = None, end = None, actions = True, auto_adjust = True, back_adjust = False):
    '''
    This function returns the historical data of a stock
    '''
    tesla = yf.Ticker(ticker)
    hist = tesla.history(period=period, interval=interval, start=start, end=end, actions=actions, auto_adjust=auto_adjust, back_adjust=back_adjust) 
    return hist

def plot_candlestick(hist, with_slider = True, support = None, resistance = None):
    '''
    This function plots the candlestick chart of a stock
    '''
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'])])
    if not with_slider:
        fig.update_layout(xaxis_rangeslider_visible=False)

    if support is not None:
        fig.add_shape(type="line",
            x0=hist.index[0],
            y0=support,
            x1=hist.index[-1],
            y1=support,
            line=dict(
                color="blue",
                width=2,
            ),
        )
        # add a text box
        fig.add_annotation(
            x=hist.index[0],
            y=support,
            text=f'Support: {support}',
            showarrow=True,
            arrowhead=1
        )

    if resistance is not None:
        fig.add_shape(type="line",
            x0=hist.index[0],
            y0=resistance,
            x1=hist.index[-1],
            y1=resistance,
            line=dict(
                color="blue",
                width=2,
            ),
        )
        # add a text box
        fig.add_annotation(
            x=hist.index[0],
            y=resistance,
            text=f'Resistance: {resistance}',
            showarrow=True,
            arrowhead=1
        )

    # add line for the current price
    fig.add_shape(type="line",
        x0=hist.index[0],
        y0=hist['Close'].iloc[-1],
        x1=hist.index[-1],
        y1=hist['Close'].iloc[-1],
        line=dict(
            color="darkgreen",
            width=2,
        ),
    )

    # add a text box
    fig.add_annotation(
        x=hist.index[-1],
        y=hist['Close'].iloc[-1],
        text=f'Current price: {hist["Close"].iloc[-1]}',
        showarrow=True,
        arrowhead=1
    )


    return fig

dictionary_period_interval = {
    '1d': '1m',
    '5d': '5m',
    '1mo': '1d',
    '3mo': '1d',
    '6mo': '1d',
    '1y': '1d',
    '2y': '1d',
    '5y': '1wk',
    '10y': '1wk',
    'ytd': '1d',
    'max': '1mo'
}


def evaluate_support_resistance(hist, verbose = True, sensibility = 5):
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
    print(ranges)


    counts = {
        f'{(bins[i]+bins[i+1])/2}': len(hist[(hist['High'] >= bins[i]) & (hist['Low'] <= bins[i+1])]) for i in range(len(bins)-1) 
    }

    print(counts)
    # keep only the strongest third dynamically
    how_many_to_keep = int(len(counts) / sensibility)
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

    return resistance, support, list_counts, current_price

def plot_support_resistance(hist, resistance = None, support = None, list_counts = None, current_price = None):
    if resistance is None or support is None or list_counts is None or current_price is None:
        resistance, support, list_counts, current_price = evaluate_support_resistance(hist, sensibility=3)
    # create a graphs plotting the hist and the support and resistance as horizontal lines
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'])])
    
    for i in list_counts:
        if i == current_price:
            fig.add_shape(type="line",
                x0=hist.index[0],
                y0=i,
                x1=hist.index[-1],
                y1=i,
                line=dict(
                    color="darkgreen",
                    width=2,
                ),
            )
            # add a text box
            fig.add_annotation(
                x=hist.index[-1],
                y=i,
                text=f'Current price: {i}',
                showarrow=True,
                arrowhead=1
            )

        elif i == resistance or i == support:
            fig.add_shape(type="line",
                x0=hist.index[0],
                y0=i,
                x1=hist.index[-1],
                y1=i,
                line=dict(
                    color="blue",
                    width=2,
                ),
            )
            # add a text box
            fig.add_annotation(
                x=hist.index[0],
                y=i,
                text=f'Support: {i}' if i == support else f'Resistance: {i}',
                showarrow=True,
                arrowhead=1
            )
        
        else:
            fig.add_shape(type="line",
                x0=hist.index[0],
                y0=i,
                x1=hist.index[-1],
                y1=i,
                line=dict(
                    color='black',
                    width=2,
                ),
            )

            # add a text box
            fig.add_annotation(
                x=hist.index[0],
                y=i,
                text=f'{i}',
                showarrow=True,
                arrowhead=1
            )


    fig.update_layout(
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    #fig.show()
    return fig
