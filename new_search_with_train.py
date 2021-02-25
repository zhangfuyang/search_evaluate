import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import cv2
import torch
from new_dataset import myDataset, trainSearchDataset
from new_scoreAgent import scoreEvaluator_with_train
import torch.nn as nn
import threading
from SVG_utils import svg_generate
from new_utils import visualization, candidate_enumerate_training, reduce_duplicate_candidate, Graph, Candidate
from colorama import Fore, Style
from dicttoxml import dicttoxml
from config import config
import torch.nn.functional as F

print(config)

def test(dataloader, model, edge_bin_size):
    model.eval()
    total_loss = {}
    batch_count = 0
    for idx, data in enumerate(dataloader):
        img = data['img'].to(model.device)
        mask = data['mask'].to(model.device)
        corner_gt_mask = data['corner_gt_mask'].to(model.device)
        gt_edge = data['gt_edge'].to(model.device).squeeze(1)
        edge_mask = data['edge_mask'].to(model.device)

        with torch.no_grad():
            img_volume = model.imgvolume(img)
            if use_bin_map:
                binmap = model.getbinmap(img)
            else:
                binmap = None
            if use_heat_map:
                heatmap = model.getheatmap(img)
                gt_heat_map = data['gt_heat_map'].to(model.device)
                heatmapMask_gt = torch.where(gt_heat_map>0.5)
                heatmapMask_pred = torch.where(heatmap>0.5)
                heatmap_l = 0
                if heatmapMask_gt[0].shape[0] > 0:
                    heatmap_l += 0.3*heatmaploss(heatmap[heatmapMask_gt], gt_heat_map[heatmapMask_gt])
                if heatmapMask_pred[0].shape[0] > 0:
                    heatmap_l += 0.3*heatmaploss(heatmap[heatmapMask_pred], gt_heat_map[heatmapMask_pred])
                heatmap_l += 0.3*heatmaploss(heatmap, gt_heat_map)

                if 'heatmap' not in total_loss.keys():
                    total_loss['heatmap'] = heatmap_l.item()
                else:
                    total_loss['heatmap'] += heatmap_l.item()
            else:
                heatmap = None

            corner_pred = model.cornerEvaluator(mask, img_volume, binmap=binmap, heatmap=heatmap)
            edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                        torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device),
                                            binmap=binmap, heatmap=heatmap)

            cornerMask_gt = torch.where(corner_gt_mask>0.5)
            cornerMask_pred = torch.where(corner_pred>0.5)
            corner_l = 0
            if cornerMask_gt[0].shape[0] > 0:
                corner_l += 0.3*cornerloss(corner_pred[cornerMask_gt], corner_gt_mask[cornerMask_gt]) 
            if cornerMask_pred[0].shape[0] > 0:
                corner_l += 0.3*cornerloss(corner_pred[cornerMask_pred], corner_gt_mask[cornerMask_pred])
            corner_l += 0.3*cornerloss(corner_pred, corner_gt_mask)

            if 'corner' not in total_loss.keys():
                total_loss['corner'] = corner_l.item()#cornerloss(corner_pred, corner_gt_mask).item()
            else:
                total_loss['corner'] += corner_l.item()#cornerloss(corner_pred, corner_gt_mask).item()
            if 'edge' not in total_loss.keys():
                total_loss['edge'] = edgeloss(edge_pred, gt_edge).item()
            else:
                total_loss['edge'] += edgeloss(edge_pred, gt_edge).item()

            if use_cross_loss:
                pseudo_corner_map = heatmap[:,1:]
                pseudo_edge_map = heatmap[:,0:1]

                # edge pseudo
                numerator = torch.mul((edge_mask+1)/2, pseudo_edge_map).sum(1).sum(1).sum(1)
                denominator = ((edge_mask+1)/2).sum(1).sum(1).sum(1)
                edge_gt_pseudo = numerator / denominator
                edge_gt_pseudo = (edge_gt_pseudo > 0.8).long()
                if 'edge_cross' not in total_loss.keys():
                    total_loss['edge_cross'] = 0.0*edgeloss(edge_pred, 1-edge_gt_pseudo).item()
                else:
                    total_loss['edge_cross'] += 0.0*edgeloss(edge_pred, 1-edge_gt_pseudo).item()

                # corner pseudo
                corner_input_mask = (mask[:, 1:]+1)/2
                corner_gt_pseudo = torch.mul(corner_input_mask, pseudo_corner_map)
                if 'corner_cross' not in total_loss.keys():
                    total_loss['corner_cross'] = cornerloss(corner_pred, 1-corner_gt_pseudo).item()
                else:
                    total_loss['corner_cross'] += cornerloss(corner_pred, 1-corner_gt_pseudo).item()

            batch_count += 1

    print('{}[TEST]{}'.format(Fore.GREEN, Style.RESET_ALL), end=' ')
    for key in total_loss.keys():
        print('{}={}'.format(key, round(total_loss[key]/batch_count, 3)), end=' ')
    print('')

    return total_loss['edge']/batch_count

def train(dataloader, model, edge_bin_size):
    model.train()
    total_loss = {}
    batch_count = 0
    for idx, data in enumerate(dataloader):
        img = data['img'].to(model.device)
        mask = data['mask'].to(model.device)
        corner_gt_mask = data['corner_gt_mask'].to(model.device)
        gt_edge = data['gt_edge'].to(model.device).squeeze(1)
        edge_mask = data['edge_mask'].to(model.device)

        optimizer.zero_grad()

        loss_dict = {}
        img_volume = model.imgvolume(img)
        if use_bin_map:
            binmap = model.getbinmap(img)
            binmap_detach = binmap
        else:
            binmap = None
            binmap_detach = None
        if use_heat_map:
            heatmap = model.getheatmap(img)
            heatmap_detach = heatmap
            gt_heat_map = data['gt_heat_map'].to(model.device)
            
            heatmapMask_gt = torch.where(gt_heat_map>0.5)
            heatmapMask_pred = torch.where(heatmap>0.5)
            heatmap_l = 0
            if heatmapMask_gt[0].shape[0] > 0:
                heatmap_l += 0.3*heatmaploss(heatmap[heatmapMask_gt], gt_heat_map[heatmapMask_gt])
            if heatmapMask_pred[0].shape[0] > 0:
                heatmap_l += 0.3*heatmaploss(heatmap[heatmapMask_pred], gt_heat_map[heatmapMask_pred])
            heatmap_l += 0.3*heatmaploss(heatmap, gt_heat_map)

            loss_dict['heatmap'] = heatmap_l

        else:
            heatmap = None
            heatmap_detach = None
        corner_pred = model.cornerEvaluator(mask, img_volume, binmap=binmap_detach, heatmap=heatmap_detach)
        edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                        torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device),
                                        binmap=binmap_detach, heatmap=heatmap_detach)

        cornerMask_gt = torch.where(corner_gt_mask>0.5)
        cornerMask_pred = torch.where(corner_pred>0.5)
        corner_l = 0
        if cornerMask_gt[0].shape[0] > 0:
            corner_l += 0.3*cornerloss(corner_pred[cornerMask_gt], corner_gt_mask[cornerMask_gt]) 
        if cornerMask_pred[0].shape[0] > 0:
            corner_l += 0.3*cornerloss(corner_pred[cornerMask_pred], corner_gt_mask[cornerMask_pred]) 
        corner_l += 0.3*cornerloss(corner_pred, corner_gt_mask)

        edge_l = edgeloss(edge_pred, gt_edge)
        loss_dict['corner'] = corner_l
        loss_dict['edge'] = edge_l
        if use_cross_loss:
            pseudo_corner_map = heatmap.detach()[:,1:]
            pseudo_edge_map = heatmap.detach()[:,0:1]

            # edge pseudo
            numerator = torch.mul((edge_mask+1)/2, pseudo_edge_map).sum(1).sum(1).sum(1)
            denominator = ((edge_mask+1)/2).sum(1).sum(1).sum(1)
            edge_gt_pseudo = numerator / denominator
            coord = data['edge_corner_coord_for_heatmap']
            index = torch.arange(coord.shape[0]).unsqueeze(1).expand(-1,2)
            pseudo_corner_score = pseudo_corner_map[index, 0, coord[index[:,0],:,0], coord[index[:,1],:,1]]
            pseudo_corner_score = pseudo_corner_score > 0.8
            pseudo_corner_score = pseudo_corner_score[:,0] & pseudo_corner_score[:,1] # two corners must all be classified as True
            edge_gt_pseudo = ((edge_gt_pseudo > 0.8) & pseudo_corner_score).long()
            loss_dict['edge_cross'] = edgeloss(edge_pred, 1-edge_gt_pseudo)

            # corner pseudo
            ### test ###
            #corner_input_mask = (mask[:, 1:]+1)/2
            #corner_gt_pseudo = torch.mul(corner_input_mask, pseudo_corner_map)
            #loss_dict['corner_cross'] = cornerloss(corner_pred, corner_gt_pseudo)

        loss = 0
        for key in loss_dict.keys():
            if key == 'corner':
                loss += loss_dict[key] * 10
            elif key == 'edge':
                loss += loss_dict[key]
            elif key == 'heatmap':
                loss += loss_dict[key] * 5
            elif key == 'edge_cross':
                loss += loss_dict[key] * 0.0 #0.1
            elif key == 'corner_cross':
                loss += loss_dict[key]

        if idx % 50 == 0:
            print('[Batch {}/{}]'.format(idx, len(dataloader)), end=' ')
            for key in loss_dict.keys():
                print('{}={}'.format(key, loss_dict[key].item()), end=' ')
            print('')

        for key in loss_dict.keys():
            if key not in total_loss.keys():
                total_loss[key] = loss_dict[key].item()
            else:
                total_loss[key] += loss_dict[key].item()
        batch_count += 1

        loss.backward()
        optimizer.step()

    print('[Overall Train]', end=' ')
    for key in total_loss.keys():
        print('{}={}'.format(key, round(total_loss[key]/batch_count, 3)), end=' ')
    print('')


class trainThread(threading.Thread):
    def __init__(self, lock, evaluator, search_evaluator, new_data_memory, dataloader, dataset, testloader):
        super(trainThread, self).__init__()
        self.lock = lock
        self.evaluator = evaluator
        self.search_evaluator = search_evaluator
        self.new_data_memory = new_data_memory
        self.dataloader = dataloader
        self.dataset = dataset
        self.testloader = testloader

    def run(self):
        print('{}[training thread]{} start'.format(Fore.BLUE, Style.RESET_ALL))
        prefix = 0
        testbest = 10
        while True:
            train_sample = 0
            for _ in range(3):
                print('{}[training thread]{} New start {}! training with {} samples'.format(
                    Fore.BLUE, Style.RESET_ALL, prefix, len(self.dataset)))
                train(self.dataloader, self.evaluator, edge_bin_size)
                self.lock.acquire()
                while len(self.new_data_memory) != 0:
                    data = self.new_data_memory.pop()
                    self.dataset.add_processed_data(data)
                # update search evaluator
                if search_with_evaluator:
                    self.search_evaluator.load_state_dict(self.evaluator.state_dict())
                self.lock.release()
                train_sample += len(self.dataset)
                #if train_sample >= MAX_DATA_STORAGE/2:
                    #break

            testacc = test(self.testloader, self.evaluator, edge_bin_size)
            print('{}[training thread]{} update searching evaluator'.format(Fore.BLUE, Style.RESET_ALL))
            print('{}[training thread]{} store weight with prefix={}'.format(Fore.BLUE, Style.RESET_ALL, prefix))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'backbone'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'backbone')))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'corner'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'corner')))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'edge'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'edge')))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'heatmapnet'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-14, 'heatmapnet')))
            self.evaluator.store_weight(save_path, prefix)
            if testacc < testbest:
                testbest = testacc
                self.evaluator.store_weight(save_path, 'best')
                with open(os.path.join(save_path, 'best.txt'), 'w') as f:
                    f.write('{} {}'.format(testbest, prefix))
            prefix += 1


class searchThread(threading.Thread):
    def __init__(self, lock, evaluator, dataloader, new_data_memory, trainDataset, searchDataset):
        super(searchThread, self).__init__()
        self.lock = lock
        self.evaluator = evaluator
        self.dataloader = dataloader
        self.new_data_memory = new_data_memory
        self.train_dataset = trainDataset
        self.search_dataset = searchDataset

    def run(self):
        print('{}[searching thread]{} start'.format(Fore.RED, Style.RESET_ALL))
        search_count = 0
        add_count = 0
        while True:
            for idx, data in enumerate(self.dataloader):
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
                        temp = random.choices(prev_candidates, k=2)

                    for candidate_ in temp:
                        corners_array = candidate_.graph.getCornersArray()
                        edges_array = candidate_.graph.getEdgesArray()
                        if corners_array.shape[0] == 0 or edges_array.shape[0] == 0:
                            continue
                        data = self.train_dataset.make_data(candidate_.name, corners_array, edges_array)
                        self.new_data_memory.append(data)
                        add_count += 1

                    for candidate_ in prev_candidates:
                        candidate_.update()  # update safe_count
                #except:
                #    print('{}[seaching thread] An error happened during searching, not sure yet, '
                #          'skip!{}'.format(Fore.RED, Style.RESET_ALL))
                #    continue

                search_count += 1
                if (idx+1) % 5 == 0:
                    print('{}[seaching thread]{} Already search {} graphs and add {} '
                          'graphs into database'.format(Fore.RED, Style.RESET_ALL, search_count, add_count))
                    print('{}[seaching thread]{} {} remain in the swap'
                          ' memory'.format(Fore.RED, Style.RESET_ALL, len(self.new_data_memory)))

             

print('process training data')
train_dataset = trainSearchDataset(config['data_folder'], data_scale=1,
                                   edge_strong_constraint=False, corner_bin=False)
#print('process testing data')
#test_dataset = trainSearchDataset(data_folder, data_scale=data_scale,
                                  #edge_strong_constraint=edge_strong_constraint, phase='valid',
                                  #corner_bin=False)

search_dataset = myDataset(config['data_folder'], phase='train', edge_linewidth=2, render_pad=-1)
'''
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)
'''
search_loader = torch.utils.data.DataLoader(search_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)

# evaluator_train is used for training
# evaluator_search is used for searching
# separate into two modules in order to use multiple threads to accelerate
#evaluator_train = scoreEvaluator_with_train(data_folder,
#                                            backbone_channel=64, edge_bin_size=edge_bin_size,
 #                                           corner_bin=False)
if config['search_with_evaluator']:
    evaluator_search = scoreEvaluator_with_train(data_folder,
                                            backbone_channel=64, edge_bin_size=edge_bin_size,
                                            corner_bin=False)
    evaluator_search.load_state_dict(evaluator_train.state_dict())
    evaluator_search.to('cuda:0')
    evaluator_search.eval()
else:
    evaluator_search = None
'''
evaluator_train.to('cuda:0')
evaluator_train.train()

optimizer = torch.optim.Adam(evaluator_train.parameters(), lr=1e-4)
cornerloss = nn.L1Loss()
heatmaploss = nn.MSELoss()
edgeloss = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1., 3.], device=evaluator_train.device))
'''
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')
######## start training and searching threads #####
lock = threading.Lock()
data_memory = []

st = searchThread(lock, evaluator_search, search_loader, data_memory, train_dataset, search_dataset)
#tt = trainThread(lock, evaluator_train, evaluator_search, data_memory, train_loader, train_dataset, test_loader)

st.run()
#tt.start()


