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
optimizer = optim.Adam(policyNet.parameters(), lr=5e-4)##optim.RMSprop(policy_net.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=0.5)

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
    def __init__(self, lock, memory, agent, policyNet, targetNet, optimizer, scheduler):
        super(trainThread, self).__init__()
        self.lock = lock
        self.memory = memory
        self.agent = agent
        self.policyNet = policyNet
        self.targetNet = targetNet
        self.optimizer = optimizer
        self.scheduler = scheduler
 
    def run(self):
        print('{}[train thread]{} start'.format(Fore.BLUE, Style.RESET_ALL))

        train_episodes = 0
        while True:
            # Return if not enough data in memory 
            if len(self.memory) < config['batch_size'] or len(self.memory) < 1000:
                time.sleep(1)
                continue 

            # Random sample a mini-batch
            transitions = self.memory.sample(config['batch_size'])

            # Compute DQN loss  
            total_loss = 0
            for tt in transitions:
                state = tt.next_state
                reward = tt.reward

                # Current q-function (policy network)
                state_action_value = self.agent.value_func(state, config['sample_edges'], self.policyNet)
                if state_action_value is None:
                    continue
                if len(state_action_value['edge_idx']) <= 1:
                    continue

                # gamma > 0
                if config['gamma'] != 0:
                    with torch.no_grad():
                        # Select action based on max_q (target network)
                        next_state_max_q = self.agent.max_q_action(state, self.targetNet)
                        if next_state_max_q.corners.shape[0] <= 1 or next_state_max_q.edges.shape[0] <= 1:
                            continue

                        # Next maximum target q-function (target network)
                        next_state_value = self.agent.value_func(next_state_max_q, config['sample_edges'], self.targetNet, 
                                            prev_state=state, prev_edge_idx=state_action_value['edge_idx'])
                        if next_state_value is None:
                            continue

                        #print(next_state_value['prev_edge_shared_idx'])  # which index outof focus edge is good  [0,1]
                        #print(state_action_value['edge_idx'])   # current state focus edge index                 [6, 8, 10]
                # gamma = 0
                else:
                    pdb.set_trace()
                    next_state_value = None
                
                loss, print_loss = agent.compute_loss(state_action_value, reward, next_state_value)                
                total_loss += loss

            total_loss /= config['batch_size']

            # Backprop and update weights 
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policyNet.parameters(), 1.0)   # clip gradient
            self.optimizer.step()
            
            if (train_episodes+1) % 1 == 0:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    break
                print('{}[train thread]{} episode: [{}] heatmap: {} corner: {} edge: {} edge-ce: {} edge_pseudo_ce {} lr {}'.format(Fore.BLUE, Style.RESET_ALL, 
                    str(train_episodes+1), round(print_loss['heatmap'],5), round(print_loss['corner'],5), 
                    round(print_loss['edge'],5), round(print_loss['edge_ce'],5), round(print_loss['edge_pseudo_ce'],5) ,lr ))
    
            if (train_episodes+1) % config['TARGET_UPDATE'] == 0:
                print('{}[train thread]{} update target network'.format(Fore.BLUE, Style.RESET_ALL))
                self.targetNet.load_state_dict(self.policyNet.state_dict())

            if (train_episodes+1) % config['SAVE_FREQ'] == 0:
                print('{}[train thread]{} save policy network'.format(Fore.BLUE, Style.RESET_ALL))
                self.policyNet.store_weight(config['save_path'], str(train_episodes+1))

            train_episodes += 1
            self.scheduler.step()

            
    
   

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

        # Keep sampling new data and insert it to memory
        total_episodes = 0
        while True:
            for data in env_dataloader:
                start_time = time.time()
                name = data['name'][0]  

                # get initial state from conv-mpn
                state = self.env.reset(name)  

                # Select action, compute reward, and save to memory 
                self.agent.select_actions(state, self.sampleNet, self.env, self.memory, config['epsilon'])  # epsilon-greedy policy + beamsearch
                total_episodes += 1

                print('{}[search thread]{} memory size: {} process time: {}'.format(Fore.RED, Style.RESET_ALL, 
                      len(self.memory), (time.time() - start_time)))

                if (total_episodes+1) % config['SAMPLE_UPDATE'] == 0:
                    print('{}[search thread]{} update sample network'.format(Fore.RED, Style.RESET_ALL))
                    self.sampleNet.load_state_dict(self.policyNet.state_dict())



lock = threading.Lock()
st = searchThread(lock, memory, env_dataloader, env, agent, sampleNet, policyNet)
tt = trainThread(lock, memory, agent, policyNet, targetNet, optimizer, scheduler)
st.start()
#tt.start()
