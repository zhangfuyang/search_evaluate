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
        print('start training')
        

        for epoch in range(config['num_epochs']):

            start_time = time.time()
            num_batch = len(self.dataset) / config['batchsize']
            for _ in range(5):
                for iteration, data in enumerate(self.dataloader):
                    img = data['img'].to(self.device)
                    target_heatmap = data['gt_heat_map'].to(self.device)
                    mask = data['mask'].to(self.device)

                    target_edgeMask = data['edge_gt_mask'].to(self.device)
                    target_cornerMask = data['corner_gt_mask'].to(self.device)

                    img_volume = self.model.imgvolume(img)  # (bs, 64, 256, 256)
                    heatmap_pred = self.model.getheatmap(img) #(bs, 2, 256, 256)
                    corner_pred = self.model.cornerEvaluator(mask.detach(), img_volume, heatmap_pred.detach()) #(bs, 1, 256, 256)
                    edge_pred = self.model.edgeEvaluator(mask.detach(), img_volume, heatmap_pred.detach()) #(bs, 2, 256, 256)
                    edge_pred_c1 = edge_pred[:,0:1]  # edge
                    edge_pred_c2 = edge_pred[:,1:]  # input mask corner

                    # Mask
                    _mask_ = mask.clone()
                    _mask_[_mask_<=0] = 0
                    mask_edge = (_mask_[:,0]>0.5).unsqueeze(1).detach()
                    mask_corner = (_mask_[:,1]>0.5).unsqueeze(1).detach()

                    # Relaxed gt labels (0.9, 0.1)
                    gt_heatmap_relax = 0.9*target_heatmap + 0.1*(1-target_heatmap)
                    gt_edge_relax = 0.9*target_edgeMask + 0.1*(1-target_edgeMask)
                    gt_corner_relax = 0.9*target_cornerMask + 0.1*(1-target_cornerMask)   
            
                    heatmap_l = heatmapLoss(heatmap_pred, gt_heatmap_relax) 
           
                    edge_l1 = 2.0*edgeLoss(edge_pred_c1*mask_edge, gt_edge_relax*mask_edge) +\
                            edgeLoss(edge_pred_c1*(~mask_edge), gt_edge_relax*(~mask_edge))

                    edge_l2 = 2.0*edgeLoss(edge_pred_c2*mask_corner, edge_pred_c2*mask_corner)

                    edge_l = edge_l1 + edge_l2
            
                    corner_l = 2.0*cornerLoss(corner_pred*mask_corner, gt_corner_relax*mask_corner) +\
                            cornerLoss(corner_pred*(~mask_corner), gt_corner_relax*(~mask_corner))

                    self.optimizer.zero_grad()
                    loss =  corner_l + heatmap_l + edge_l
                    loss.backward()
                    self.optimizer.step()
       
                    if iteration % config['print_freq'] == 0:
                        print('[Epoch %d: %d/%d] corner loss: %.4f, edge loss: %.4f, heatmap loss: %.4f, total loss: %.4f' % 
                            (epoch, iteration+1, num_batch, corner_l.item(), edge_l.item(), heatmap_l.item(), loss.item()))

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

           
            self.lock.acquire()
            while len(self.data_memory) != 0:
                data = self.data_memory.pop()
                self.dataset._add_processed_data_(data)
            self.lock.release()
           
 
         

   
class searchThreadRL(threading.Thread):

    def __init__(self, search_dataset, search_dataloader, train_dataset, model, data_memory):
        super(searchThreadRL, self).__init__()
        self.search_dataset = search_dataset
        self.search_dataloader = search_dataloader
        self.train_dataset = train_dataset
        self.model = model
        self.data_memory = data_memory

    def run(self):
        """data augmentation."""
        print('{}[searching thread]{} start'.format(Fore.RED, Style.RESET_ALL))
        search_count = 0
        add_count = 0
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

                # Initial start state (from conv-mpn)
                initial_candidate = Candidate.initial(Graph(corners, edges), name)
                prev_candidates = [initial_candidate]

                for epoch_i in range(config['beam_depth']):
                    updated_candidates = []

                    for prev_i in range(len(prev_candidates)):
                        prev_ = prev_candidates[prev_i]
                        if len(prev_.graph.getCorners()) == 0 or len(prev_.graph.getEdges()) == 0:
                            continue
                        
                        # Extend an action 
                        next_candidates = candidate_enumerate_training(prev_, gt_data)
                        next_candidates = reduce_duplicate_candidate(next_candidates)

                        # Sort by scores
                        self.model.get_score_list(next_candidates) 
                        next_candidates = sorted(next_candidates, key=lambda x:x.graph.graph_score(), reverse=True)

                        # select top-k candidates
                        top_k_next_candidates = next_candidates[:min(config['beam_width'], len(next_candidates))]

                        # epsilon-greedy policy
                        selected_next_candidates = []
                        for top_k in top_k_next_candidates:
                            if random.random() > config['epsilon']:
                                selected_next_candidates.append(top_k)
                            else:
                                selected_next_candidates.append(random.sample(next_candidates, 1)[0])

                        if epoch_i == 0:
                            updated_candidates += random.sample(selected_next_candidates, min(len(selected_next_candidates), config['beam_width']))
                        else:
                            updated_candidates += random.sample(selected_next_candidates, min(len(selected_next_candidates), 2))
                
                    # Store  in memory 
                    temp = random.sample(updated_candidates, min(len(updated_candidates), 1))
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
                        if ratio > 0.2 and ratio < 0.8:
                            self.data_memory.append(data)
                            add_count += 1

                    # set next state as current state         
                    prev_candidates = updated_candidates
         


   
