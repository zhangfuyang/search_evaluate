import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import copy
import torch
from new_config import config
from new_scoreAgent import scoreEvaluator_with_train as DQN
from agent import ReplayMemory, Agent
from new_dataset import myDataset, trainSearchDataset 
from env import BuildingEnv
from itertools import count
from dicttoxml import dicttoxml

import pdb

# Save config file 
print(config)
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')

# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
policyNet_device = torch.device("cuda:0")
targetNet_device = torch.device("cuda:1")

# Initialize network
policyNet = DQN(os.path.join(config['base_path'], 'cities_dataset'),
                backbone_channel=64, edge_bin_size=config['edge_bin_size'],
                corner_bin=False)
policyNet.to(policyNet_device)
policyNet.train()

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
ground_truth_dataset = trainSearchDataset(config['data_folder'], data_scale=config['data_scale'],
                                   edge_strong_constraint=config['edge_strong_constraint'], corner_bin=False)

# Initialize agent and environment
agent = Agent(policyNet, targetNet, ground_truth_dataset) 
env = BuildingEnv(env_dataset, ground_truth_dataset)


#########################
# Optimize for one step #
#########################
def optimize_model():
    # Return if not enough data in memory 
    if len(memory) < config['batch_size']:
        return 
    # Random sample a mini-batch
    transitions = memory.sample(config['batch_size'])

    total_loss = 0
    for tt in transitions:
        state = tt.next_state
        reward = tt.reward

        # current q-function (policy network)
        state_action_value = agent.value_func(state, config['sample_edges'], use_policy_net=True)

        if len(state_action_value['edge_idx']) <= 0:
            return None

        # select action based on max_q (target network)
        next_state_max_q = agent.max_q_action(state, use_target_net=True)

        if next_state_max_q.corners.shape[0] <= 1:
            return None

        if config['gamma'] != 0:
            # next maximum target q-function (target network)
            next_state_value = agent.value_func(next_state_max_q, config['sample_edges'], use_policy_net=False, 
                                            prev_state=state, prev_edge_idx=state_action_value['edge_idx'])
        else:
            next_state_value = None
        # Compute the DQN loss 
        loss = agent.compute_loss(state_action_value, reward, next_state_value)
        total_loss += loss

    total_loss /= config['batch_size']

    # Backprop and update weights 
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(policyNet.parameters(), 1.0)   # clip gradient
    optimizer.step()
    
    return total_loss.item() 
    

######################
# Main training loop #
######################
epsilon = config['epsilon']
total_episodes = 0
train_episodes = 0

for i_episode in range(config['num_episodes']):
    for data in env_dataloader:
        print('episode: [%d/%d]' % (total_episodes, len(env_dataloader)*config['num_episodes']))
        name = data['name'][0]  # get img name 
        state = env.reset(name)  # reset env from this img

        do_train = len(memory) > config['MEMORY_SIZE']/5
        if do_train:
            train_episodes += 1

        # Run one episode 
        for t in count(): 
            # Select action and perform an action (policy network)
            next_state = agent.select_action(state, epsilon, use_target_net=False)  # epsilon-greedy policy + transition
            if next_state.corners.shape[0] <= 1:
                break

            # Get rewards from environment (next_state = current state + action)
            rewards, done = env.step(next_state)  
            
            # Store transition in memory
            memory.push(next_state, rewards) 

            # Perform one step of the optimization (on the target network)
            if do_train:
                loss = optimize_model()
                if loss is not None:
                    print('loss at step %d: %.3f' % (t, loss))
            
            # Move to the next state 
            if done or t >= config['max_run']:
                break  # done or spend too much time
            else:
                state = copy.deepcopy(next_state)
            
   
        # Update the target network, copying all weights and biases in DQN
        if train_episodes % config['TARGET_UPDATE'] == 0 and do_train:
            print('update target network...')
            targetNet.load_state_dict(policyNet.state_dict())
        
        total_episodes += 1

    print('saving weights')
    policyNet.store_weight(config['save_path'], str(train_episodes))
        
