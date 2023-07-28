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

class RL_Agent:
    def __init__(self):
        self.model, self.trainer = self.create_model()
        self.memory = deque(maxlen=100_000)
        self.model_name = 'Strategy_1Agent'

    def create_model(self):
        # now feed each day of february and calculate the resistance and support
        number_of_inputs = 52
        number_of_hidden_neurons = 500
        number_of_outputs = 3
        # try to load the model
        try:
            model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
            model.load()
        except:
            model = Linear_QNet(number_of_inputs, number_of_hidden_neurons, number_of_outputs)
        trainer = QTrainer(model, 0.001, 0.9)
        return model, trainer
    
    def restart_model(self):
        import os
        # remove the folder and the file
        os.remove('model/model.pth')
        os.rmdir('model')
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # save model
        self.model.save()
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_state_option_1(self,i, hist_complete):
        i = i-1
        # current price is the close of the previous day
        new_price = hist_complete.iloc[i]['Close']
        average = hist_complete.iloc[:i]['Close'].mean()
        average = [average , new_price]

        hist_for_support_resistance = hist_complete.iloc[:100+i]
        hist_for_support_resistance = hist_for_support_resistance[:100]
        
        resistance, support, list_counts, current_price, list_resistances, list_supports = evaluate_support_resistance_for_ML(hist_for_support_resistance, verbose=False, sensibility=5)
        supports_and_resistances = list_counts

        try:
            index_current_price = supports_and_resistances.index(new_price)
        except:
            # get the closest price
            index_current_price = min(supports_and_resistances, key=lambda x:abs(x-new_price))
            index_current_price = supports_and_resistances.index(index_current_price)
        list_of_zeros = [i-current_price for i in supports_and_resistances]
        # at the index of the current price we put 1
        list_of_zeros[index_current_price] = 1
        state = list_of_zeros + average
        return state
    
    def get_state_option_2(self,i, hist_complete):
        i = i-1
        if i >= 50:
            # current price is the close of the previous day
            hist_till_today = hist_complete.iloc[:i]
            new_price = hist_complete.iloc[i]['Close']
            average = hist_complete.iloc[:i]['Close'].mean()
            average = [average , new_price]
            hist_for_support_resistance = hist_till_today.iloc[-50:]
            hist_for_support_resistance = hist_for_support_resistance['Close'].tolist()
            state = hist_for_support_resistance[-50:] + average
            return state
        else:
            return [0 for i in range(52)]
    
    def get_reward(self,state, action, next_state):
        '''
        Options: tied to the current price and the next price
        let's suppose we have this:
        state =  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        action = 0 (buy)
        next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        reward = 1
        '''
        current_price = state[-1]
        next_price = next_state[-1]
        if action == 0: # buy
            if next_price > current_price:
                return 1
            elif next_price < current_price:
                return -1
            else:
                return 0
        elif action == 1: # sell
            # find index of 1 in state
            if next_price < current_price:
                return 1
            elif next_price > current_price:
                return -1
            else:
                return 0
        elif action == 2: # hold
            return 0

    def _get_random_action(self,with_string=False):
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
            
    def get_next_state(self,i, option=1, hist = None):
        if option == 1:
            try:
                state = self.get_state_option_1(i+1, hist)
            except:
                state = self.get_state_option_1(i, hist)
        elif option == 2:
            try:
                state = self.get_state_option_2(i+1, hist)
            except:
                state = self.get_state_option_2(i, hist)
        return state
    
    def get_done(self, i, score, hist_complete):
        '''
        if finish row return true otherwise return false
        '''
        if i == len(hist_complete) - 1 or score < 0:
            done = False
        else:
            done = True
        return done

    def get_action_from_state(self, state, with_string=False):
        random_number_1_to_100 = random.randint(1, 100)
        # if the random number is less than 10 we choose a random action
        if random_number_1_to_100 < 10 and with_string:
            out, out_string = self._get_random_action(with_string=True)
            return out, out_string
        else:
            # here we are using the model to get the action
            classes = ['buy', 'sell', 'hold']
            state = torch.tensor(state, dtype=torch.float)
            tensor_out = self.model.predict(state)
            # normalize the tensor
            tensor_out = torch.softmax(tensor_out, dim=0)
            # st.write(tensor_out)
            # st.write(torch.argmax(tensor_out))
            # st.write(classes[torch.argmax(tensor_out)])
            # # write sum of tensor
            # st.write(torch.sum(tensor_out))
            if torch.sum(tensor_out) >1.1 or torch.sum(tensor_out) < 0.9:
                st.warning('The sum of the tensor is not 1 but {}'.format(torch.sum(tensor_out)))
                st.stop()
            else: 
                # if the confidence il less than 0.5 we choose a random action
                if torch.max(tensor_out) < 0.4 and with_string:
                    out, out_string = self._get_random_action(with_string=True)
                    return out, out_string
                else:
                    out = torch.argmax(tensor_out)
                    if with_string:
                        return out, classes[out]
                    return out

    def step_agent(self, hist, i=1, score=0):
        hist_until_today = hist.iloc[0:i]
        state = self.get_state_option_2(i, hist_complete = hist)
        action, action_string = self.get_action_from_state(state, with_string=True)
        next_state = self.get_next_state(i, option=2, hist = hist)
        reward = self.get_reward(state, action, next_state)
        done = self.get_done(i, score, hist)
        self.train_short_memory( state, action, reward, next_state, done)
        self.remember(state, action, reward, next_state, done)                        
        if done or i == len(hist_complete) - 1:
            self.train_long_memory()
        return action_string, state, i
