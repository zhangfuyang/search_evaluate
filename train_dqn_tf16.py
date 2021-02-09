import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
import random
import copy
import cv2
import torch
from new_config import config
from new_scoreAgent import scoreEvaluator_with_train as DQN
from agent import ReplayMemory, Agent
from new_dataset import myDataset, trainSearchDataset 
from env import BuildingEnv
from itertools import count

import pdb

print(config)

# Initialize cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
policy_net_device = torch.device("cuda:0")
target_net_device = torch.device("cuda:0")

# Initialize policy network
policy_net = DQN(os.path.join(config['base_path'], 'cities_dataset'),
                                            backbone_channel=64, edge_bin_size=config['edge_bin_size'],
                                            corner_bin=False)
policy_net.to(policy_net_device)
policy_net.train()

# Initialize target network
target_net = DQN(os.path.join(config['base_path'], 'cities_dataset'),
                                            backbone_channel=64, edge_bin_size=config['edge_bin_size'],
                                            corner_bin=False)
target_net.load_state_dict(policy_net.state_dict())
target_net.to(target_net_device)
target_net.eval()

# Initialize optimizer
optimizer = optim.RMSprop(policy_net.parameters())

# Initialize replay memory
memory = ReplayMemory(config['MEMORY_SIZE'])

# Initialize data loader 
env_dataset = myDataset(config['data_folder'], phase='train', edge_linewidth=2, render_pad=-1)
env_dataloader = torch.utils.data.DataLoader(env_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)
ground_truth_dataset = trainSearchDataset(config['data_folder'], data_scale=config['data_scale'],
                                   edge_strong_constraint=config['edge_strong_constraint'], corner_bin=False)

# Initialize environment 
env = BuildingEnv(env_dataset, ground_truth_dataset)

# Initialize agent
agent = Agent(policy_net, target_net, ground_truth_dataset)

scaler = torch.cuda.amp.GradScaler()

#########################
# optimize for one step #
#########################
def optimize_model():
    # Return if not enough data in memory 
    if len(memory) < config['batch_size']:
        return 
    # Random sample a mini-batch
    transitions = memory.sample(config['batch_size'])


    total_loss = 0
    with torch.cuda.amp.autocast():
        for tt in transitions:
            state = tt.next_state
            reward = tt.reward
            # current q-function (corner & edge scores)
            state_action_value = agent.value_func(state, config['sample_edges'], use_policy_net=True)

            # select action based on max_q (target network)
            next_state_max_q = agent.max_q_action(state, use_target_net=True)

            # next maximum target q-function (same corner & edge scores)
            next_state_value = agent.value_func(next_state_max_q, config['sample_edges'], use_policy_net=True, 
                                            prev_state=state, prev_edge_idx=state_action_value['edge_idx'])
        
            # Compute the DQN loss 
            loss = agent.compute_loss(state_action_value, next_state_value, reward)
            total_loss += loss

        total_loss /= config['batch_size']

    # Backprop and update weights 
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)   # clip gradient
    scaler.step(optimizer)
    scaler.update()
    
    return total_loss.item() 
    

#################
# training loop #
#################
epsilon = config['epsilon']
total_episodes = 0
for i_episode in range(config['num_episodes']):
    for data in env_dataloader:
        print('##############')
        print('episode: ',total_episodes)
        name = data['name'][0]  # get img name 
        state = env.reset(name)  # reset env from this img

        # Run episode 
        for t in count(): 
            # Select action and perform an action (policy network)
            next_state = agent.select_action(state, epsilon, use_target_net=False)  # epsilon-greedy policy + stepping
            if next_state.corners.shape[0] <= 1:
                break
            
            # Get rewards
            rewards, done = env.step(next_state)  
            
            # Store transition in memory
            memory.push(next_state, rewards) 

            # Perform one step of the optimization (on the target network)
            loss = optimize_model()
            if loss is not None:
                print('loss at step %d: %.3f' % (t, loss))
            
            # Move to the next state 
            if done or t >= config['max_run']:
                break  # done or spend too much time
            else:
                state = copy.deepcopy(next_state)
            
   
        # Update the target network, copying all weights and biases in DQN
        if total_episodes % config['TARGET_UPDATE'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        total_episodes += 1
        

