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

# create a reinforcement learning model that takes a state and return an action
# the state is a list of resistances and supports and current prices and return a action (buy, sell, hold)

'''
supports_and_resistances = [11,11.5,12,12.5,13,13.5,14,14.5,15,15.5]
current_price = 12.5
capital = 1000

state = [[supports_and_resistances], current_price]

action = model.predict(state)

if action == 'buy':
    # create a buy order

if action == 'sell':
    # create a sell order

if action == 'hold':

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from utils import *

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx].item()  # Convert the tensor to a scalar using .item()
            if not done[idx]:
                Q_new = reward[idx].item() + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

if __name__ == '__main__':
    # Paramenters of the model
    number_of_inputs = 11
    number_of_hidden_neurons = 256
    number_of_outputs = 3

    model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
    trainer = QTrainer(model, 0.001, 0.9)
    
    def get_state():
        '''
        The state will contain a list of supports and resistances and the current price.
        '''
        state = [1,2,3,4,5,6,7,8,9,10,11]
        return state
    
    def get_reward():
        '''
        if position is closed return the reward otherwise return 0
        '''
        return 1
    
    def get_action():
        '''
        the action is a buy or sell or hold
        '''
        return [1,0,0]
    
    def get_next_state():
        '''
        the next state is the state after the action
        '''
        # get the next state from the data
        next_state = [1,2,3,4,5,6,7,8,9,10,11]
        return next_state
    
    def get_done():
        done = [0]
        return done
    
    def training_loop(epochs):
        for i in range(epochs):
            state = get_state()
            action = get_action()
            reward = get_reward()
            next_state = get_next_state()
            done = get_done()
            trainer.train_step(state, action, reward, next_state, done)
            print(i)
            # save the model at the end of the training
            if i == epochs - 1:
                model.save()

    #training_loop(1000)


    '''
    The idea is that we can simulate a stock trading environment and train the model on that environment.

    We need metaphors for the following:
    - state
    - action
    - reward
    - next_state
    - done

    The state will contain a list of supports and resistances and the current price.
    - the state will be a list of 11 prices

    The action is a buy or sell or hold
    - it will create a buy or sell order setting the stop loss and take profit to one of the supports or resistances depending on the action
    - it will close the position if the stop loss or take profit is hit

    The reward is the profit or loss of the position
    - if position is closed return the reward otherwise return 0
    - the reward will be the profit or loss of the position


    The next state is the state after the action
    - the next resistance or support plus the new price

    The done is a boolean that is true if the position is closed otherwise is false
    - if the position is closed return true otherwise return false

    The trainer will train the model on the state, action, reward, next_state, done
    - the trainer will train the model on the state, action, reward, next_state, done

    '''
    # show the choce of the model
    # I want to give batch of stock prices and get the action to do

    ticker = 'eurusd=x'
    period = '1d'
    interval = '1m'

    eur_usd = yf.Ticker(ticker)
    hist = eur_usd.history(period=period, interval=interval)
    hist = hist.dropna()


    from utils import evaluate_support_resistance

    resistance, support, list_counts, current_price = evaluate_support_resistance(hist, verbose=False, sensibility=5)
    # we can interpret them as 0, and 1
    # the 0s are the supports and the the 1 is the current price (or the resistance)
    # - encode (1,0,-1) -> supports, current price, resistance



    '''
    def check_if_close(position):
        if position['status'] == 'open':
            if position['take_profit'] <= position['current_price'] or position['stop_loss'] >= position['current_price']:
                position['status'] = 'closed'
                position['profit'] = position['current_price'] - position['open_price'] if position['type_of_position'] == 'buy' else position['open_price'] - position['current_price']
        return position

    def get_reward(position):
        if position['status'] == 'closed':
            return position['profit']
        else:
            return 0

    def open_position_buy():
        price = get_price()
        position = {
            'status': 'open',
            'type_of_position': 'buy',
            'open_price': price,
            'take_profit': price + 0.01,
            'stop_loss': price - 0.01,
            'current_price': price,
            'profit': 0
        }
        return position

    def open_position_sell():
        price = get_price()
        position = {
            'status': 'open',
            'type_of_position': 'sell',
            'open_price': price,
            'take_profit': price - 0.01,
            'stop_loss': price + 0.01,
            'current_price': price,
            'profit': 0
        }
        return position


                '''
    print(list_counts)
