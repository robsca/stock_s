'''
Create a strategy for the model
as a first we feed the data.
What's the question?
What's the answer?

Predictor 
- we need to evaluate if is a good moment to buy or sell or hold ( we cannot know the answer ) 
- We need to create a classifier that identify the most predictable trends
- Use RL and ML to train the model on trading the best strategies when identify the trends.

Risk Control
- We need to use statistics to evaluate the risk of the operation
- We need to use statistics to evaluate the risk of the portfolio


Simulation Gym
- Testing strategy on Historical Data

How to manage risk?
- We can use correlation to better have a better control:

    example. 
    stock 1 is highly inversly correlated to stock 2.

    capital: 100


    Open a position of buy with stock 1,

    

Need a training gym for the algorithms
--- 

let's say that if we hit a resistance and 50 percent of the time 
- we buy with TP to next support and SL to next resistance
- we sell with SL to next resistance and TP to next Support


1. Evaluate a support 
2. Evaluate a resistance
3. Create a order 
4. Calculate the result

'''

from utils import *


def evaluate_support_resistance(hist, verbose = True):
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
    bins = np.linspace(min_price, max_price, 10)
    # from the bins create the ranges 
    ranges = {
        f'{(bins[i]+bins[i+1])/2}': [bins[i], bins[i+1]] for i in range(len(bins)-1) for j in range(1)
    }
    print(ranges)


    counts = {
        f'{(bins[i]+bins[i+1])/2}': len(hist[(hist['High'] >= bins[i]) & (hist['Low'] <= bins[i+1])]) for i in range(len(bins)-1) 
    }

    print(counts)

    # keep only the strongest half
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
    counts = dict(list(counts.items())[len(counts)//2:])


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

def plot_support_resistance(hist):
    resistance, support, list_counts, current_price = evaluate_support_resistance(hist)
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
                    color="LightSeaGreen",
                    width=2,
                ),
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

    fig.update_layout(
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    fig.show()

async def main():
    hist = await get_stock_price(period = '1d', interval = '15m', time_to_sleep = 'dead')
    print(hist)
    while True: 
        hist = await get_stock_price(period = '1d', interval = '15m', time_to_sleep = 1)
        evaluation = evaluate_support_resistance(hist)

if __name__ == "__main__":
    asyncio.run(main())