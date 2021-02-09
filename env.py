from collections import namedtuple
import numpy as np
import random
import os
import torch
from new_utils import remove_intersection_and_duplicate, \
    sort_graph, render, get_wrong_corners, get_wrong_edges, simplify_gt, get_corner_bin_map
from new_config import config

import pdb 

#########################
# definiation for state #
#########################
State = namedtuple('State', ['name', 'corners', 'edges'])

###############
# environment #
###############
class BuildingEnv():

    def __init__(self, env_dataset, ground_truth):
        self.env_dataset = env_dataset
        self.ground_truth = ground_truth


    def reset(self, name):
        graph_data = self.env_dataset.getDataByName(name)
        conv_data = graph_data['conv_data']
        corners = conv_data['corners']
        corners = np.round(corners).astype(np.int)
        edges = conv_data['edges']
        state = State(name, corners, edges)
        return state 


    def compute_rewards(self, state, corner_false_id, edge_false_id):
        edges = state.edges
        corners = state.corners
        
        ### corner reward ###
        ### reward of 1 for correct corner ###
        corner_correct_id = list(set(range(corners.shape[0])) - set(corner_false_id))
        corner_gt = render(corners[corner_correct_id], np.array([]), render_pad=0, scale=config['data_scale'])[1]
        
        ###  edge reward  ###
        ### reward of 1 for correct edge ###
        edge_gt = []
        for id_ in range(edges.shape[0]):
            edge_label = [0 if id_ in edge_false_id else 1]
            edge_gt.append(edge_label[0])
        edge_gt = np.array(edge_gt)
            
        ### region ###
        # direct learn segmentation from image, not include in the searching system
        # TODO: could add here as well
        return corner_gt, edge_gt

    
    def compute_false_id(self, state):
        corners = state.corners
        edges = state.edges

        gt_data = self.ground_truth.ground_truth[state.name]
        
        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            corners, gt_data['corners'], edges, gt_data['edges'])

        gt_corners, gt_edges = simplify_gt(map_same_location, gt_data['corners'], gt_data['edges'])

        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            corners, gt_corners, edges, gt_edges)

        if self.ground_truth.edge_strong_constraint:
            edge_false_id = get_wrong_edges(
                corners, gt_corners, edges, gt_edges,
                map_same_degree)
        else:
            edge_false_id = get_wrong_edges(
                corners, gt_corners, edges, gt_edges,
                map_same_location)
        corner_list_for_each_bin = None
        corner_false_id = list(corner_false_id)

        return corner_false_id, edge_false_id


    def step(self, state):
        corner_false_id, edge_false_id = self.compute_false_id(state)
        corner_gt, edge_gt = self.compute_rewards(state, corner_false_id, edge_false_id)

        rewards={}
        rewards['corner_gt'] = corner_gt
        rewards['edge_gt'] = edge_gt
        rewards['corner_false_id'] = corner_false_id
        rewards['edge_false_id'] = edge_false_id

        done = (len(corner_false_id) + len(edge_false_id) == 0)
        
        return rewards, done
   





