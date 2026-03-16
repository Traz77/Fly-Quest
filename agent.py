from collections import deque
import random 
import torch
import torch.nn as nn
import torch.optim as optim
from network import DQN

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space # ["w", "a", "s", "d"]
    
        self.epsilon = 1 # Drunk factor
        self.epsilon_decay = 0.995
        
        self.gamma = 0.9 # Future actions preffered 90% of the time 

        self.memory = deque(maxlen=10000) # Acts as a short term memory
        
        self.brain = DQN()
        self.target_brain = DQN()
        self.train_step_counter = 0
        self.target_brain.load_state_dict(self.brain.state_dict())

        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.batch_size = 64

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        # Ask for prediction and turn the tuple to tensor object
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            q_values = self.brain(state_tensor)

        best_action_index = torch.argmax(q_values).item()

        return self.action_space[best_action_index]

    def train(self, state, action, reward, next_state, done):

        # Wait until enough memory for a batch size 
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([exp[0] for exp in batch])

        # Turn text to numbers 
        action_indices = [self.action_space.index(exp[1]) for exp in batch]
        actions = torch.LongTensor(action_indices).unsqueeze(1)

        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])

        # What did the brain orginally predict for these 64 states 
        curent_q_values = self.brain(states).gather(1, actions).squeeze()

        # What should it have predicted based on the rewards we got - do not update weights based on future
        max_future_q_values = self.target_brain(next_states).max(1)[0].detach()

        # Target Q = Reward + (Gamma * Future Q). if game ends - future is 0
        target_q_values = rewards + (self.gamma * max_future_q_values * (1 - dones))

        # MSELoss 
        loss = self.criterion(curent_q_values, target_q_values)

        # Zero any old data from last step 
        self.optimizer.zero_grad()

        # Gradiants calculaiton 
        loss.backward()

        # Update weights 
        self.optimizer.step()
        self.train_step_counter += 1

        if self.train_step_counter % 100 == 0:
            self.target_brain.load_state_dict(self.brain.state_dict())

    def remember(self, state, action, reward, next_state, done):
           self.memory.append((state, action, reward, next_state, done))
