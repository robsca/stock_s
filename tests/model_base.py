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
import random 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from scripts.utils import *

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = x 

        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))

    def predict(self, state):
        return self.forward(state)

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
    # try to load the model
    try:
        model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
        model.load()
    except:
        model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
    trainer = QTrainer(model, 0.001, 0.9)
    
    def get_state():
        '''
        This will be a series of 11 digits, they represent 10 supports and resistances and the current price
        The support and resistances are 0s, the current price is 1
        '''
        # initial state is 11 zeros
        state = [0 for i in range(11)]
        # set the current price to 1
        index_current_price = random.randint(0, len(state) - 1)
        state[index_current_price] = 1
        return state
    
    def get_reward(state, action, next_state):
        '''
        Options: tied to the current price and the next price
        let's suppose we have this:
        state =  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        action = 0 (buy)
        next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        reward = 1
        '''
        if action == 0:
            if state[-1] == 1 and next_state[-1] == 1:
                return 1
            return -1
        if action == 1:
            if state[-1] == 1 and next_state[-1] == 1:
                return 1
            return -1
        if action == 2:
            if state[-1] == 1 and next_state[-1] == 1:
                return 1
            return -1
        return 0
        
    def get_action(with_string=False):
        '''
        At the moment is a simple random choice between the 3 options.
        
        Goal
        ---
        Create a order buy, or sell
        '''
        classes = ['buy', 'sell', 'hold']
        indexes = [0,1,2]
        action = random.choice(indexes)
        if with_string:
            return action, classes[action]
        return action
    
    def get_action_from_state(state, with_string=False):
        # here we are using the model to get the action
        # tranform in tensor
        state = torch.tensor(state, dtype=torch.float)
        if with_string:
            return torch.argmax(model(state)).item(), get_action(with_string=True)
        return torch.argmax(model(state)).item()
    
    def get_next_state():
        state = get_state()
        return state
    
    def get_done():
        done = [0]
        return done
    
    score = 0
    def training_loop(epochs):
        global score
        for i in range(epochs):
            state = get_state()
            action, action_string = get_action_from_state(state, with_string=True)
            next_state = get_next_state()
            reward = get_reward(state, action, next_state)
            score = score + reward
            done = get_done()
            trainer.train_step(state, action, reward, next_state, done)
            print('Action: ', action_string)
            print('Reward: ', reward)
            print('Next State: ', next_state)
            print('Done: ', done)
            print('Score: ', score)
            print(i)
            # save the model at the end of the training
            if i == epochs - 1:
                model.save()

        print('Model Score: ', score)

    training_loop(1000)