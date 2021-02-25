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

import pdb

cornerLoss = torch.nn.L1Loss()
edgeLoss = torch.nn.L1Loss()
heatmapLoss = torch.nn.MSELoss()

def trainThread(dataset, dataloader, model, optimizer, scheduler, device, shared_buffer, lock):
    """Main training code."""
    # Add new graph from buffer to training dataset
    lock.acquire()
    while len(shared_buffer) != 0:
        data = shared_buffer.pop()
        dataset.add_processed_data(data)
    lock.release()
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        num_batch = len(dataset) / config['batchsize']

        for iteration, data in enumerate(dataloader):
            img = data['img'].to(device)
            target_heatmap = data['gt_heat_map'].to(device)
            mask = data['mask'].to(device)

            target_edgeMask = data['edge_gt_mask'].to(device)
            target_cornerMask = data['corner_gt_mask'].to(device)

            img_volume = model.imgvolume(img)  # (bs, 64, 256, 256)
            heatmap_pred = model.getheatmap(img) #(bs, 2, 256, 256)
            corner_pred = model.cornerEvaluator(mask.detach(), img_volume, heatmap_pred.detach()) #(bs, 1, 256, 256)
            edge_pred = model.edgeEvaluator(mask.detach(), img_volume, heatmap_pred.detach()) #(bs, 2, 256, 256)
            #edge_pred_c1 = edge_pred[:,0:1,:,:]
            #edge_pred_c2 = edge_pred[:,1:,:,:]
            

            # Mask
            _mask_ = mask.clone()
            _mask_[_mask_<=0] = 0
            mask_edge = (_mask_[:,0]>0.5).unsqueeze(1).detach()
            mask_corner = (_mask_[:,1]>0.5).unsqueeze(1).detach()

            # Relaxed gt labels (0.9, 0.1)
            gt_heatmap_relax = 0.9*target_heatmap + 0.1*(1-target_heatmap)
            gt_edge_relax = 0.9*target_edgeMask + 0.1*(1-target_edgeMask)
            gt_corner_relax = 0.9*target_cornerMask + 0.1*(1-target_cornerMask)   
            
            #corner_l = cornerLoss(corner_pred, target_cornerMask)
            #edge_l = edgeLoss(edge_pred, target_edgeMask)
            #heatmap_l = heatmapLoss(heatmap_pred, target_heatmap)

            heatmap_l = heatmapLoss(heatmap_pred, gt_heatmap_relax) 
           
            edge_l = 2.0*edgeLoss(edge_pred*mask_edge, gt_edge_relax*mask_edge) +\
                         edgeLoss(edge_pred*(~mask_edge), gt_edge_relax*(~mask_edge))

            #edge_l2 = 2.0*edgeLoss(edge_pred_c2*mask_corner, gt_edge_relax*mask_corner)
            
            corner_l = 2.0*cornerLoss(corner_pred*mask_corner, gt_corner_relax*mask_corner) +\
                           cornerLoss(corner_pred*(~mask_corner), gt_corner_relax*(~mask_corner))
            #edge_l = edge_l1 + edge_l2

            optimizer.zero_grad()
            loss =  corner_l + heatmap_l + edge_l
            loss.backward()
            optimizer.step()
       
            if (iteration+1) % config['print_freq'] == 0:
                print('[Epoch %d: %d/%d] corner loss: %.4f, edge loss: %.4f, heatmap loss: %.4f, total loss: %.4f' % 
                    (epoch, iteration+1, num_batch, corner_l.item(), edge_l.item(), heatmap_l.item(), loss.item()))
        
        # LR decay
        scheduler.step()

        print("--- %s seconds for one epoch ---" % (time.time() - start_time))
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        print('Learning Rate: %s' % str(lr))
        start_time = time.time()

        # Save model weights
        if (epoch+1) % config['save_freq'] == 0:
            model.store_weight(config['save_path'], str(epoch+1))



   
def searchThread(search_dataset, search_dataloader, train_dataset, shared_buffer):
    """data augmentation."""
    print('{}[searching thread]{} start'.format(Fore.RED, Style.RESET_ALL))
    search_count = 0
    add_count = 0
    while True:
        for idx, data in enumerate(search_dataloader):
            # Load gt graph 
            name = data['name'][0]
            graph_data = search_dataset.getDataByName(name)
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
                current_candidates = []
                for prev_i in range(len(prev_candidates)):
                    prev_ = prev_candidates[prev_i]
                    if len(prev_.graph.getCorners()) == 0 or len(prev_.graph.getEdges()) == 0:
                        continue
                    current_candidates.extend(candidate_enumerate_training(prev_, gt_data))

                if len(current_candidates) == 0:
                    break
                current_candidates = reduce_duplicate_candidate(current_candidates)

                if True:
                    if len(current_candidates) > config['beam_width']:
                        prev_candidates = random.sample(current_candidates, config['beam_width'])
                    else:
                        prev_candidates = current_candidates
                    temp = random.choices(prev_candidates, k=1)

                for candidate_ in temp:
                    corners_array = candidate_.graph.getCornersArray()
                    edges_array = candidate_.graph.getEdgesArray()
                    if corners_array.shape[0] == 0 or edges_array.shape[0] == 0:
                        continue

                    # Add data to buffer
                    data = train_dataset.make_data(candidate_.name, corners_array, edges_array)
                    # Avoid all false or all empty bias case
                    ratio = len(data['edge_false_id']) / data['edges'].shape[0]
                    if ratio > 0.2 and ratio < 0.8:
                        shared_buffer.append(data)
                        add_count += 1

                for candidate_ in prev_candidates:
                    candidate_.update()  # update safe_count
               
            search_count += 1
            if (idx+1) % 1 == 0:
                print('{}[seaching thread]{} Already search {} graphs and add {} '
                        'graphs into database'.format(Fore.RED, Style.RESET_ALL, search_count, add_count))






train_dataset = EvaluatorDataset(config['data_folder'], phase='train',  edge_strong_constraint=False)
gt_datapath = train_dataset.gt_datapath
conv_mpn_datapath = train_dataset.conv_mpn_datapath


def searchThread_preprocess(name, shared_buffer):
    """data augmentation."""
    gt_data = np.load(os.path.join(gt_datapath, name+'.npy'), allow_pickle=True).tolist()
    gt_data['corners'], gt_data['edges'] = sort_graph(gt_data['corners'], gt_data['edges'])
    gt_data['corners'] = np.round(gt_data['corners']).astype(np.int)

    convmpn_data = np.load(os.path.join(conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
    convmpn_data['corners'], convmpn_data['edges'] = \
        remove_intersection_and_duplicate(convmpn_data['corners'], convmpn_data['edges'], name)
    convmpn_data['corners'], convmpn_data['edges'] = sort_graph(convmpn_data['corners'], convmpn_data['edges'])
    corners = convmpn_data['corners']
    corners = np.round(corners).astype(np.int)
    edges = convmpn_data['edges']

    # Initial start state (from conv-mpn)
    initial_candidate = Candidate.initial(Graph(corners, edges), name)
    prev_candidates = [initial_candidate]

    for epoch_i in range(config['beam_depth']):
        current_candidates = []
        for prev_i in range(len(prev_candidates)):
            prev_ = prev_candidates[prev_i]
            if len(prev_.graph.getCorners()) == 0 or len(prev_.graph.getEdges()) == 0:
                continue
            current_candidates.extend(candidate_enumerate_training(prev_, gt_data))

        if len(current_candidates) == 0:
            break
        current_candidates = reduce_duplicate_candidate(current_candidates)

        if len(current_candidates) > config['beam_width']:
            prev_candidates = random.sample(current_candidates, config['beam_width'])
        else:
            prev_candidates = current_candidates
        temp = random.choices(prev_candidates, k=1)

        for candidate_ in temp:
            corners_array = candidate_.graph.getCornersArray()
            edges_array = candidate_.graph.getEdgesArray()
            if corners_array.shape[0] == 0 or edges_array.shape[0] == 0:
                continue

            # Add data to buffer
            data = train_dataset.make_data(candidate_.name, corners_array, edges_array)
            # Avoid all false or all empty bias case
            ratio = len(data['edge_false_id']) / data['edges'].shape[0]
            if ratio > 0.2 and ratio < 0.8:
                shared_buffer.append(data)

        for candidate_ in prev_candidates:
            candidate_.update()  # update safe_count

               
   
