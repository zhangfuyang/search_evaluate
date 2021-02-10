from collections import namedtuple
import numpy as np
import random
import torch
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from new_utils import candidate_enumerate_training, reduce_duplicate_candidate, Graph, Candidate
from env import State
from new_utils import remove_intersection_and_duplicate, \
    sort_graph, render, get_wrong_corners, get_wrong_edges, simplify_gt, get_corner_bin_map
import matplotlib.pyplot as plt
from new_config import config
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import torch.nn as nn

import pdb

##################################
# definiation for one transition #
##################################
Transition = namedtuple('Transition',('next_state', 'reward'))


#################
# replay memory #
#################
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
      
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


############
# RL agent #
############
class Agent():

    def __init__(self, policy_net, target_net, ground_truth):
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.policy_net = policy_net
        self.target_net = target_net
        self.ground_truth = ground_truth

    def select_action(self, state, epsilon, use_target_net=True):
        if use_target_net:
            model = self.target_net
        else:
            model = self.policy_net
        # rank value function from a set of heurastic actions 
        with torch.no_grad():
            initial_candidate = Candidate.initial(Graph(state.corners, state.edges), state.name)
            next_candidates = candidate_enumerate_training(initial_candidate, 1)
            next_candidates = reduce_duplicate_candidate(next_candidates)
            model.get_score_list(next_candidates, all_edge=True)
            next_candidates = sorted(next_candidates, key=lambda x:x.graph.graph_score(), reverse=True) # descending scores
        
        # epsilon-greedy policy
        if random.random() > epsilon:
            next_candidate = next_candidates[0]
        else:
            next_candidate = random.sample(next_candidates, 1)[0]
        
        next_corners = next_candidate.graph.getCornersArray()
        next_edges = next_candidate.graph.getEdgesArray()
        next_state = State(state.name, next_corners, next_edges)
        return next_state 

    def max_q_action(self, state, use_target_net=True):
        next_state = self.select_action(state, epsilon=-1, use_target_net=use_target_net)
        return next_state

    def value_func(self, state, num_edges, use_policy_net=True, prev_state=None, prev_edge_idx=None):
        if use_policy_net:
            model = self.policy_net
        else:
            model = self.target_net

        # Image feature 
        img = skimage.img_as_float(plt.imread(os.path.join(config['data_folder'], 'rgb', state.name+'.jpg')))
        img = img.transpose((2,0,1))
        img = (img - np.array(config['mean'])[:, np.newaxis, np.newaxis]) / np.array(config['std'])[:, np.newaxis, np.newaxis]
        img = torch.FloatTensor(img).unsqueeze(0).to(model.device)
        img_volume = model.imgvolume(img)

        # Heatmap
        binmap = None
        binmap_detach = None
        if config['use_heat_map']:
            with torch.no_grad():
                heatmap = model.getheatmap(img)
                heatmap_detach = heatmap
        else:
            heatmap = None
            heatmap_detach = None

        
        # Full mask
        mask = render(state.corners, state.edges, render_pad=-1, scale=config['data_scale'])
        mask = torch.FloatTensor(mask).unsqueeze(0).to(model.device)

        edge_masks = []
        prev_edge_shared_idx = []

        # Use previous state edge locations
        if prev_edge_idx is not None:
            assert prev_state is not None
            # Find previous selected edges that exists in current edges 
            old_corner_id = prev_state.edges[prev_edge_idx]
            old_corner_pos = prev_state.corners[old_corner_id]

            for idx, old_pos in enumerate(old_corner_pos):
                # Test if edge corner has been removed 
                has_remove = False
                new_id = np.array([-1,-1])
                for corner_i in range(2):
                    point = old_pos[corner_i]
                    l2_d = np.linalg.norm(point-state.corners, axis=1)
                    if np.min(l2_d) >= 6:
                        has_remove = True
                        break
                    new_id[corner_i] = np.argmin(l2_d)

                # Test if edge corners is detected 
                edge_exist = False
                if not has_remove:
                    if new_id[0] in list(state.edges[:,0]) and new_id[1] in list(state.edges[:,1]):
                        edge_exist = True
                    if new_id[0] in list(state.edges[:,1]) and new_id[1] in list(state.edges[:,0]):
                        edge_exist = True
                
                # Render a shared edge 
                if not has_remove and edge_exist:
                    edge_mask = render(state.corners, new_id[np.newaxis,:], render_pad=-1, scale=config['data_scale'])[0]
                    edge_masks.append(edge_mask)
                    prev_edge_shared_idx.append(idx)

        # Random sampled edge masks
        else:
            if num_edges >= state.edges.shape[0]:
                edge_idx = list(range(0, state.edges.shape[0]))
            else:
                edge_idx = random.sample(range(0, state.edges.shape[0]), num_edges) 
            
            for id_ in edge_idx:
                edge_mask = render(state.corners, state.edges[[id_]], render_pad=-1, scale=config['data_scale'])[0]
                edge_masks.append(edge_mask)

        edge_masks = np.array(edge_masks)
        edge_masks = torch.FloatTensor(edge_masks).unsqueeze(1).to(model.device)
        assert(edge_masks.shape[0] > 0)
        
        # corner score
        corner_pred = model.cornerEvaluator(mask, img_volume, binmap=binmap_detach, heatmap=heatmap_detach)
        
        # edge score 
        expand_shape = edge_masks.shape[0]
        bin_map_extend = None
        if config['use_heat_map']:
            heatmap_extend = heatmap_detach.expand(expand_shape,-1,-1,-1)
        else:
            heatmap_extend = None
        edge_batch_pred = model.edgeEvaluator(
                    edge_masks,
                    mask.expand(expand_shape,-1,-1,-1),
                    img_volume.expand(expand_shape,-1,-1,-1),
                    corner_pred.expand(expand_shape,-1,-1,-1),
                    torch.zeros(expand_shape, config['edge_bin_size'], device=model.device),
                    binmap=bin_map_extend,
                    heatmap=heatmap_extend)
        
        out_data = {}
        out_data['corner_pred'] = corner_pred
        out_data['edge_batch_pred'] = edge_batch_pred
        if prev_edge_idx is None:
            out_data['edge_idx'] = edge_idx
        else:
            out_data['edge_idx'] = None
        out_data['prev_edge_shared_idx'] = prev_edge_shared_idx
        out_data['heatmap'] = heatmap
        out_data['name'] = state.name
        return out_data

    def compute_loss(self, state_value, reward, next_state_value=None):
        if next_state_value is not None:
            # Find removed and shared Edge
            assert state_value['edge_batch_pred'].shape[0] == len(state_value['edge_idx'])
            assert len(state_value['edge_idx']) >= len(next_state_value['prev_edge_shared_idx']) 
            removed_edges = set(np.arange(len(state_value['edge_idx']))) - set(next_state_value['prev_edge_shared_idx'])
            if len(removed_edges) == 0:
                removed_edges = []
            else:
                removed_edges = list(removed_edges)
            shared_edges = next_state_value['prev_edge_shared_idx']
            assert len(state_value['edge_idx']) == len(shared_edges) + len(removed_edges) 
        
            prev_edges_value = state_value['edge_batch_pred']
            next_edges_value = next_state_value['edge_batch_pred']
            next_edges_value_ = torch.zeros_like(prev_edges_value).to(prev_edges_value.device)
            next_edges_value_[shared_edges] = next_edges_value.detach().cpu().to(prev_edges_value.device)
            next_edges_value_[removed_edges] = 0.5  # Special case: removed edge
            edge_rewards = torch.FloatTensor(reward['edge_gt'][state_value['edge_idx']]).to(prev_edges_value.device).unsqueeze(1)
        
            # reward + discounted return 
            expected_edge_values = (next_edges_value_ * config['gamma']) + edge_rewards * (1-config['gamma'])
      
            # Edge loss 
            edge_loss = F.smooth_l1_loss(prev_edges_value, expected_edge_values.detach())
        
            # Corner loss
            prev_corner_value = state_value['corner_pred']
            corner_rewards = torch.FloatTensor(reward['corner_gt']).to(prev_corner_value.device).reshape(1,1,256,256)
            gt_mask = (corner_rewards>0).detach()
            next_corner_value = next_state_value['corner_pred'].detach().cpu().to(prev_corner_value.device) * gt_mask.reshape(1,1,256,256)
            expected_corner_values = (next_corner_value * config['gamma']) + corner_rewards * (1-config['gamma'])
            corner_loss = F.smooth_l1_loss(prev_corner_value, expected_corner_values.detach())

        # No value function 
        else:
            # Edge loss 
            prev_edges_value = state_value['edge_batch_pred']
            edge_rewards = torch.FloatTensor(reward['edge_gt'][state_value['edge_idx']]).to(prev_edges_value.device).unsqueeze(1)
            edge_loss = F.smooth_l1_loss(prev_edges_value, edge_rewards.detach())

            # Corner loss
            prev_corner_value = state_value['corner_pred']
            corner_rewards = torch.FloatTensor(reward['corner_gt']).to(prev_corner_value.device).reshape(1,1,256,256)
            corner_loss = F.smooth_l1_loss(prev_corner_value, corner_rewards.detach())

        # Heatmap loss 
        heatmap = state_value['heatmap'][0]
        gt_data = self.ground_truth.ground_truth[state_value['name']]
        gt_corners = gt_data['corners']
        gt_edges = gt_data['edges']
        gt_heat_map = render(gt_corners, gt_edges, render_pad=0, corner_size=5, edge_linewidth=3)
        gt_heat_map = torch.FloatTensor(gt_heat_map).to(heatmap.device).detach()
        heatmaploss = nn.MSELoss()
        heatmap_l = heatmaploss(heatmap, gt_heat_map)
        
        return edge_loss + corner_loss*10.0 + heatmap_l*5.0
