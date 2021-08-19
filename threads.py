import random 
from colorama import Fore, Style
from new_utils import visualization, candidate_enumerate_training, reduce_duplicate_candidate, Graph, Candidate
import threading
import numpy as np
import os
from new_utils import remove_intersection_and_duplicate, sort_graph
from config import config 
import time 
import torch 
import torch.nn.functional as F
from multiprocessing import Manager
import matplotlib.pyplot as plt
from new_dataset import EvaluatorDataset
from new_scoreAgent import *
import pdb
import threading

cornerLoss = torch.nn.L1Loss()
edgeLoss = torch.nn.L1Loss()
heatmapLoss = torch.nn.MSELoss()



class trainThread(threading.Thread):

    def __init__(self, train_dataset, train_dataloader, model, optimizer, scheduler, device, searchModel, lock, data_memory):
        super(trainThread, self).__init__()
        self.dataset = train_dataset
        self.dataloader = train_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lock = lock
        self.searchModel = searchModel
        self.data_memory = data_memory


    def run(self):

        """Main training code."""
        print('{}[train thread]{} start'.format(Fore.BLUE, Style.RESET_ALL))
        
        for epoch in range(config['num_epochs']):
            
            if epoch > 0:
                # Wait for search thread if too fast
                while len(self.data_memory) <= 120*config['graph_per_data']:  
                    print('waiting for search...')
                    print(len(self.data_memory))
                    time.sleep(120)
                print(len(self.data_memory))
                self.lock.acquire()
                while len(self.data_memory) != 0:
                    data = self.data_memory.pop()
                    self.dataset._add_processed_data_(data)
                self.lock.release()
            
            start_time = time.time()
            num_batch = len(self.dataset) / config['batchsize']
            for iteration, data in enumerate(self.dataloader):
                img = data['img'].to(self.device)
                target_heatmap = data['gt_heat_map'].to(self.device)
                mask = data['mask'].to(self.device)

                target_edgeMask = data['edge_gt_mask'].to(self.device)
                target_cornerMask = data['corner_gt_mask'].to(self.device)

                heatmap_pred = self.model.getheatmap(img) 

                img_volume = self.model.imgvolume(img) 
                corner_pred = self.model.cornerEvaluator(mask.detach(), img_volume, heatmap_pred) 
                edge_pred = self.model.edgeEvaluator(mask.detach(), img_volume, heatmap_pred) 
                    
                _mask_ = mask.clone()
                _mask_[_mask_<=0] = 0
                mask_edge = (_mask_[:,0]>0.5).unsqueeze(1).detach()
                mask_corner = (_mask_[:,1]>0.5).unsqueeze(1).detach()
            
                heatmap_l = heatmapLoss(heatmap_pred, target_heatmap)
                                    
                edge_l = 2.0*F.smooth_l1_loss(edge_pred*mask_edge, target_edgeMask*mask_edge, beta=0.5) +\
                                F.smooth_l1_loss(edge_pred*(~mask_edge), target_edgeMask*(~mask_edge), beta=0.5) 

                corner_l = 2.0*F.smooth_l1_loss(corner_pred*mask_corner, target_cornerMask*mask_corner, beta=0.5) +\
                                F.smooth_l1_loss(corner_pred*(~mask_corner), target_cornerMask*(~mask_corner), beta=0.5)

                self.optimizer.zero_grad()
                loss =  corner_l + edge_l + heatmap_l
                loss.backward()
                self.optimizer.step()
       
                if iteration % config['print_freq'] == 0:
                    print('[Epoch %d: %d/%d] heat loss: %.4f, corner loss: %.4f, edge loss: %.4f, total loss: %.4f' % 
                        (epoch, iteration+1, num_batch, heatmap_l.item(), edge_l.item(), corner_l.item(), loss.item()))
                
                # Update search model
                self.lock.acquire()
                self.searchModel.load_state_dict(self.model.state_dict())
                self.lock.release()
                
            # Save model weights
            if (epoch+1) % config['save_freq'] == 0:
                self.model.store_weight(config['save_path'], str(epoch+1))
        
            # LR decay
            self.scheduler.step()

            print("--- %s seconds for one epoch ---" % (time.time() - start_time))
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            print('Learning Rate: %s' % str(lr))
            start_time = time.time()
            
            

   
class searchThread(threading.Thread):

    def __init__(self, search_dataset, search_dataloader, train_dataset, model, data_memory):
        super(searchThread, self).__init__()
        self.search_dataset = search_dataset
        self.search_dataloader = search_dataloader
        self.train_dataset = train_dataset
        self.model = model
        self.data_memory = data_memory


    def run(self):
        """data augmentation."""
        print('{}[search thread]{} start'.format(Fore.RED, Style.RESET_ALL))
 
        while True:
            for idx, data in enumerate(self.search_dataloader):
                # Load gt graph 
                name = data['name'][0]
                graph_data = self.search_dataset.getDataByName(name)
                conv_data = graph_data['conv_data']
                corners = conv_data['corners']
                corners = np.round(corners).astype(np.int)
                edges = conv_data['edges']

                gt_data = graph_data['gt_data']
                gt_corners = gt_data['corners']
                gt_corners = np.round(gt_corners).astype(np.int)
                gt_edges = gt_data['edges']

                # Initial start state 
                initial_candidate = Candidate.initial(Graph(corners, edges), name)
                prev_candidates = [initial_candidate]

                sampled_data = []
                for epoch_i in range(config['beam_depth']):
                    current_candidates = []

                    # Enumerate candidates
                    for prev_i in range(len(prev_candidates)):
                        prev_ = prev_candidates[prev_i]
                        if len(prev_.graph.getCorners()) == 0 or len(prev_.graph.getEdges()) == 0:
                            continue
                        # Extend graph 
                        current_candidates.extend(candidate_enumerate_training(prev_, gt_data))
                    
                    # Reduce duplicate
                    current_candidates = reduce_duplicate_candidate(current_candidates)

                    # Sort by scores
                    with torch.no_grad():
                        self.model.get_score_list(current_candidates)

                    # Select top-k candidates
                    current_candidates = sorted(current_candidates, key=lambda x:x.graph.graph_score(), reverse=True)
                    top_k_next_candidates = current_candidates[:min(config['beam_width'], len(current_candidates))]

                    
                    # epsilon-greedy policy
                    selected_next_candidates = []
                    for top_k in top_k_next_candidates:
                        if random.random() > config['epsilon']:
                            selected_next_candidates.append(top_k)
                        else:
                            selected_next_candidates.append(random.sample(current_candidates, 1)[0])
                    
                    sampled_data += selected_next_candidates
                    
                    # set next state as current state         
                    prev_candidates = selected_next_candidates

                # Store  in memory 
                temp = sampled_data 
                
                for idx_, candidate_ in enumerate(temp):
                    # create next states:
                    corners_array = candidate_.graph.getCornersArray()
                    edges_array = candidate_.graph.getEdgesArray()
                       
                    if corners_array.shape[0] == 0 or edges_array.shape[0] == 0:
                        continue
                        
                    # Add data to buffer
                    data = self.train_dataset.make_data(candidate_.name, corners_array, edges_array)
                    # Avoid all false or all empty bias case
                    ratio = len(data['edge_false_id']) / data['edges'].shape[0]
                    if ratio > 0.3:
                        self.data_memory.append(data)

            
