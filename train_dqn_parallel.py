import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import copy
import torch
import time 
from new_config import config
from new_scoreAgent import scoreEvaluator_with_train as DQN
from agent import ReplayMemory, Agent
from new_dataset import myDataset, trainSearchDataset 
from env import BuildingEnv, State
from itertools import count
from dicttoxml import dicttoxml
import threading
from colorama import Fore, Style
import random

import pdb

# Save config file 
print(config)
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')

# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
policyNet_device = torch.device("cuda:0")
targetNet_device = torch.device("cuda:0")
sampleNet_device = torch.device("cuda:0")

# Initialize network
policyNet = DQN(os.path.join(config['base_path'], 'cities_dataset'),
                backbone_channel=64, edge_bin_size=config['edge_bin_size'],
                corner_bin=False)
policyNet.to(policyNet_device)

sampleNet = DQN(os.path.join(config['base_path'], 'cities_dataset'),
                backbone_channel=64, edge_bin_size=config['edge_bin_size'],
                corner_bin=False)
sampleNet.load_state_dict(policyNet.state_dict())
sampleNet.to(sampleNet_device)

targetNet = DQN(os.path.join(config['base_path'], 'cities_dataset'),
                backbone_channel=64, edge_bin_size=config['edge_bin_size'],
                corner_bin=False)
targetNet.load_state_dict(policyNet.state_dict())
targetNet.to(targetNet_device)
targetNet.eval()


# Initialize optimizer
optimizer = optim.Adam(policyNet.parameters(), lr=1e-4)##optim.RMSprop(policy_net.parameters())

# Initialize replay memory
memory = ReplayMemory(config['MEMORY_SIZE'])

# Initialize data loader 
env_dataset = myDataset(config['data_folder'], phase='train', edge_linewidth=2, render_pad=-1)
env_dataloader = torch.utils.data.DataLoader(env_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)
ground_truth_dataset = trainSearchDataset(config['data_folder'], phase='train', data_scale=config['data_scale'],
                                   edge_strong_constraint=config['edge_strong_constraint'], corner_bin=False)

# Initialize agent and environment
agent = Agent(env_dataset, ground_truth_dataset) 
env = BuildingEnv(env_dataset, ground_truth_dataset)

print(len(env_dataset))

class trainThread(threading.Thread):
    def __init__(self, lock, memory, agent, policyNet, targetNet, optimizer):
        super(trainThread, self).__init__()
        self.lock = lock
        self.memory = memory
        self.agent = agent
        self.policyNet = policyNet
        self.targetNet = targetNet
        self.optimizer = optimizer
 
    def run(self):
        print('{}[train thread]{} start'.format(Fore.BLUE, Style.RESET_ALL))

        train_episodes = 0
        while True:
            # Return if not enough data in memory 
            if len(self.memory) < config['batch_size'] or len(self.memory) < config['MEMORY_SIZE']/5:
                time.sleep(5)
                continue 

            # Random sample a mini-batch
            transitions = self.memory.sample(config['batch_size'])
            
            total_loss = 0
            for tt in transitions:
                state = tt.next_state
                reward = tt.reward

                # current q-function (policy network)
                state_action_value = self.agent.value_func(state, config['sample_edges'], self.policyNet)
                if len(state_action_value['edge_idx']) <= 0:
                    continue

                # select action based on max_q (target network)
                next_state_max_q = self.agent.max_q_action(state, self.targetNet)
                if next_state_max_q.corners.shape[0] <= 1:
                    continue

                if config['gamma'] != 0:
                    # next maximum target q-function (target network)
                    next_state_value = self.agent.value_func(next_state_max_q, config['sample_edges'], self.targetNet, 
                                            prev_state=state, prev_edge_idx=state_action_value['edge_idx'])
                else:
                    next_state_value = None
                # Compute the DQN loss 
                loss, print_loss = agent.compute_loss(state_action_value, reward, next_state_value)
                total_loss += loss
            total_loss /= config['batch_size']

            # Backprop and update weights 
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policyNet.parameters(), 1.0)   # clip gradient
            self.optimizer.step()

            if (train_episodes+1) % 10 == 0:
                print('{}[train thread]{} episode: [{}] heatmap: {} corner: {} edge: {} edge-xe: {}'.format(Fore.BLUE, Style.RESET_ALL, 
                    str(train_episodes+1), round(print_loss['heatmap'],5), round(print_loss['corner'],5), 
                    round(print_loss['edge'],5), round(print_loss['edge_ce'],5) ))
            train_episodes += 1

            if (train_episodes+1) % config['TARGET_UPDATE'] == 0:
                print('{}[train thread]{} update target network'.format(Fore.BLUE, Style.RESET_ALL))
                #self.lock.acquire()
                self.targetNet.load_state_dict(self.policyNet.state_dict())
                #self.lock.release()

            if (train_episodes+1) % config['SAVE_FREQ'] == 0:
                print('{}[train thread]{} save policy network'.format(Fore.BLUE, Style.RESET_ALL))
                self.policyNet.store_weight(config['save_path'], str(train_episodes+1))
    
   

class searchThread(threading.Thread):
    def __init__(self, lock, memory, env_dataloader, env, agent, sampleNet, policyNet):
        super(searchThread, self).__init__()
        self.lock = lock
        self.env_dataloader = env_dataloader
        self.env = env
        self.memory = memory
        self.agent = agent
        self.sampleNet = sampleNet
        self.policyNet = policyNet

    def run(self):
        print('{}[search thread]{} start'.format(Fore.RED, Style.RESET_ALL))
        epsilon = config['epsilon']

        total_episodes = 0
        # Keep sampling new data and insert it to memory
        while True:
            for data in env_dataloader:
                print('{}[search thread]{} episode: [{}] memory size: {}'.format(Fore.RED, Style.RESET_ALL, 
                      total_episodes, len(self.memory)))
                name = data['name'][0]  # get img name 

                # choose between conv-mpn or gt initial state
                graph_data = env_dataset.getDataByName(name)
                gt_data = graph_data['gt_data']
                gt_corners = gt_data['corners']
                gt_edges = gt_data['edges']
                if random.random() > 0.8:
                    state = State(name, gt_corners, gt_edges)
                else:
                    state = self.env.reset(name)  # reset env from this img

                # Run one episode 
                for t in count(): 
                    # Select action and perform an action (policy network)
                    next_state = self.agent.select_action(state, epsilon, self.sampleNet)  # epsilon-greedy policy + transition
                    if next_state.corners.shape[0] <= 1:
                        break

                    # Get rewards from environment (next_state = current state + action)
                    rewards, done = self.env.step(next_state)  
            
                    # Store transition in memory
                    self.memory.push(next_state, rewards) 

                    # Move to the next state 
                    if done or t >= config['max_run']:
                        break  # done or spend too much time
                    else:
                        state = copy.deepcopy(next_state)
            
                # Update the target network, copying all weights and biases in DQN
                if (total_episodes+1) % config['SAMPLE_UPDATE'] == 0:
                    print('{}[search thread]{} update sample network'.format(Fore.RED, Style.RESET_ALL))
                    self.sampleNet.load_state_dict(self.policyNet.state_dict())
        
                total_episodes += 1


lock = threading.Lock()
st = searchThread(lock, memory, env_dataloader, env, agent, sampleNet, policyNet)
tt = trainThread(lock, memory, agent, policyNet, targetNet, optimizer)
st.start()
tt.start()