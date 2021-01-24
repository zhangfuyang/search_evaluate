import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from new_config import *

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
                heatmap_l = heatmaploss(heatmap, gt_heat_map)
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
            if 'corner' not in total_loss.keys():
                total_loss['corner'] = cornerloss(corner_pred, corner_gt_mask).item()
            else:
                total_loss['corner'] += cornerloss(corner_pred, corner_gt_mask).item()
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
                    total_loss['edge_cross'] = edgeloss(edge_pred, edge_gt_pseudo).item()
                else:
                    total_loss['edge_cross'] += edgeloss(edge_pred, edge_gt_pseudo).item()

                # corner pseudo
                corner_input_mask = (mask[:, 1:]+1)/2
                corner_gt_pseudo = torch.mul(corner_input_mask, pseudo_corner_map)
                if 'corner_cross' not in total_loss.keys():
                    total_loss['corner_cross'] = cornerloss(corner_pred, corner_gt_pseudo).item()
                else:
                    total_loss['corner_cross'] += cornerloss(corner_pred, corner_gt_pseudo).item()

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
            binmap_detach = binmap.detach()
        else:
            binmap = None
            binmap_detach = None
        if use_heat_map:
            heatmap = model.getheatmap(img)
            heatmap_detach = heatmap.detach()
            gt_heat_map = data['gt_heat_map'].to(model.device)
            heatmap_l = heatmaploss(heatmap, gt_heat_map)
            loss_dict['heatmap'] = heatmap_l
        else:
            heatmap = None
            heatmap_detach = None
        corner_pred = model.cornerEvaluator(mask, img_volume, binmap=binmap_detach, heatmap=heatmap_detach)
        edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                        torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device),
                                        binmap=binmap_detach, heatmap=heatmap_detach)
        corner_l = cornerloss(corner_pred, corner_gt_mask)
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
            edge_gt_pseudo = (edge_gt_pseudo > 0.8).long()
            loss_dict['edge_cross'] = edgeloss(edge_pred, edge_gt_pseudo)

            # corner pseudo
            corner_input_mask = (mask[:, 1:]+1)/2
            corner_gt_pseudo = torch.mul(corner_input_mask, pseudo_corner_map)
            loss_dict['corner_cross'] = cornerloss(corner_pred, corner_gt_pseudo)

        loss = 0
        for key in loss_dict.keys():
            if key == 'corner':
                loss += loss_dict[key] * 10
            elif key == 'edge':
                loss += loss_dict[key]
            elif key == 'heatmap':
                loss += loss_dict[key] * 5
            elif key == 'edge_cross':
                loss += loss_dict[key] * 0.5
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
                self.lock.release()
                train_sample += len(self.dataset)
                if train_sample >= MAX_DATA_STORAGE-1:
                    break

            testacc = test(self.testloader, self.evaluator, edge_bin_size)
            print('{}[training thread]{} update searching evaluator'.format(Fore.BLUE, Style.RESET_ALL))
            print('{}[training thread]{} store weight with prefix={}'.format(Fore.BLUE, Style.RESET_ALL, prefix))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-7, 'backbone'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-7, 'backbone')))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-7, 'corner'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-7, 'corner')))
            if os.path.exists(os.path.join(save_path, '{}_{}.pt'.format(prefix-7, 'edge'))):
                os.remove(os.path.join(save_path, '{}_{}.pt'.format(prefix-7, 'edge')))
            self.evaluator.store_weight(save_path, prefix)
            prefix += 1
            if testacc < testbest:
                testbest = testacc
                self.evaluator.store_weight(save_path, 'best')
                with open(os.path.join(save_path, 'best.txt'), 'w') as f:
                    f.write('{} {}'.format(testbest, prefix))


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

                #try:
                initial_candidate = Candidate.initial(Graph(corners, edges), name)
                #self.lock.acquire()
                #self.evaluator.get_score(initial_candidate)
                #self.lock.release()

                prev_candidates = [initial_candidate]

                for epoch_i in range(beam_depth):
                    current_candidates = []
                    for prev_i in range(len(prev_candidates)):
                        prev_ = prev_candidates[prev_i]
                        if len(prev_.graph.getCorners()) == 0 or len(prev_.graph.getEdges()) == 0:
                            continue
                        current_candidates.extend(candidate_enumerate_training(prev_, gt_data))

                    if len(current_candidates) == 0:
                        break
                    current_candidates = reduce_duplicate_candidate(current_candidates)


                    #for candidate_ in current_candidates:
                    #    self.lock.acquire()
                    #    self.evaluator.get_score(candidate_, all_edge=True)
                    #    self.lock.release()

                    #for candidate_i in range(len(current_candidates)):
                    #    if best_candidates[0].graph.graph_score() < current_candidates[candidate_i].graph.graph_score():
                    #        best_candidates = [current_candidates[candidate_i]]
                    #    elif best_candidates[0].graph.graph_score() == current_candidates[candidate_i].graph.graph_score():
                    #        best_candidates.append(current_candidates[candidate_i])

                    #current_candidates = sorted(current_candidates, key=lambda x:x.graph.graph_score(), reverse=True)
                    #if len(current_candidates) < beam_width:
                    #    pick = np.arange(len(current_candidates))
                    #else:
                    #    pick = np.arange(beam_width)

                    #prev_candidates = [current_candidates[_] for _ in pick]

                    if len(current_candidates) > beam_width:
                        prev_candidates = random.sample(current_candidates, beam_width)
                    else:
                        prev_candidates = current_candidates

                    # add random two into training dataset
                    temp = random.choices(prev_candidates, k=2)
                    for candidate_ in temp:
                        corners_array = candidate_.graph.getCornersArray()
                        edges_array = candidate_.graph.getEdgesArray()
                        if corners_array.shape[0] == 0 or edges_array.shape[0] == 0:
                            continue
                        data = self.train_dataset.make_data(candidate_.name, corners_array, edges_array)
                        self.lock.acquire()
                        self.new_data_memory.append(data)
                        self.lock.release()
                        add_count += 1

                    for candidate_ in prev_candidates:
                        candidate_.update() # update safe_count
                #except:
                #    print('{}[seaching thread] An error happened during searching, not sure yet, '
                #          'skip!{}'.format(Fore.RED, Style.RESET_ALL))
                #    continue


                search_count += 1
                if (idx+1) % 100 == 0:
                    print('{}[seaching thread]{} Already search {} graphs and add {} '
                          'graphs into database'.format(Fore.RED, Style.RESET_ALL, search_count, add_count))
                    print('{}[seaching thread]{} {} remain in the swap'
                          ' memory'.format(Fore.RED, Style.RESET_ALL, len(self.new_data_memory)))

                #if save_count <= 20:
                #    save_count += 1
                #    print('{}[seaching thread]{} Save sample {} into {} with '
                #          'depth={}'.format(Fore.RED, Style.RESET_ALL, name,
                #                            str(self.curr_state['save_count'])+'_checkpoint',len(candidate_gallery)-1))
                #    gt_candidate = Candidate.initial(Graph(gt_corners, gt_edges), name)
                #    self.evaluator.get_score(gt_candidate)
                #    save_gallery(candidate_gallery, name,
                #                 os.path.join(save_path, 'search_demo', str(self.curr_state['save_count'])+'_checkpoint'),
                #                 best_candidates, gt_candidate)

def save_candidate_image(candidate, base_path, base_name):
    corners = candidate.graph.getCornersArray()
    edges = candidate.graph.getEdgesArray()
    # graph svg
    svg = svg_generate(corners, edges, base_name, samecolor=True)
    svg.saveas(os.path.join(base_path, base_name+'.svg'))
    # corner image
    temp_mask = np.zeros((256,256))
    for ele in candidate.graph.getCorners():
        if ele.get_score() < 0:
            temp_mask = cv2.circle(temp_mask, ele.x[::-1], 3, 1, -1)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1,1)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name+'_corner.png'), dpi=256)
    # edges image
    temp_mask = np.zeros((256,256))
    for ele in candidate.graph.getEdges():
        if ele.get_score() < 0:
            A = ele.x[0]
            B = ele.x[1]
            temp_mask = cv2.line(temp_mask, A.x[::-1], B.x[::-1], 1, thickness=1)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name+'_edge.png'), dpi=256)
    # region no need fig
    plt.close()

def save_gallery(candidate_gallery, name, save_path, best_candidates, gt_candidate):
    os.makedirs(save_path, exist_ok=True)
    base_path = os.path.join(save_path, name)
    os.makedirs(base_path, exist_ok=True)

    ##################################### GT ############################################
    base_name = 'gt_pred'
    save_candidate_image(gt_candidate, base_path, base_name)

    ################################### search ##########################################
    for k in range(len(candidate_gallery)):
        current_candidates = candidate_gallery[k]
        for idx, candidate_ in enumerate(current_candidates):
            base_name = 'iter_'+str(k)+'_num_'+str(idx)
            save_candidate_image(candidate_, base_path, base_name)

    #################################### best ###########################################
    for k in range(len(best_candidates)):
        candidate_ = best_candidates[k]
        base_name = 'best_'+str(k)
        save_candidate_image(candidate_, base_path, base_name)

    ################################ save config ########################################
    data = {}
    # gt
    corner_count = 0
    edge_count = 0
    for ele in gt_candidate.graph.getCorners():
        if ele.get_score() < 0:
            corner_count +=1
    for ele in gt_candidate.graph.getEdges():
        if ele.get_score() < 0:
            edge_count += 1
    data['gt'] = {'score': round(gt_candidate.graph.graph_score(), 2),
                  'corner_score': round(gt_candidate.graph.corner_score(), 2),
                  'edge_score': round(gt_candidate.graph.edge_score(), 2),
                  'region_score': round(gt_candidate.graph.region_score(), 2),
                  'false_corner': corner_count,
                  'false_edge': edge_count}

    # pred
    for k in range(len(candidate_gallery)):
        current_candidates = candidate_gallery[k]
        for idx, candidate_ in enumerate(current_candidates):
            corner_count = 0
            edge_count = 0
            for ele in candidate_.graph.getCorners():
                if ele.get_score() < 0:
                    corner_count +=1
            for ele in candidate_.graph.getEdges():
                if ele.get_score() < 0:
                    edge_count += 1
            data['iter_{}_num_{}'.format(k, idx)] = {'score': round(candidate_.graph.graph_score(), 2),
                                                     'corner_score': round(candidate_.graph.corner_score(), 2),
                                                     'edge_score': round(candidate_.graph.edge_score(), 2),
                                                     'region_score': round(candidate_.graph.region_score(), 2),
                                                     'false_corner': corner_count,
                                                     'false_edge': edge_count}

    # best
    for idx, candidate_ in enumerate(best_candidates):
        corner_count = 0
        edge_count = 0
        for ele in candidate_.graph.getCorners():
            if ele.get_score() < 0:
                corner_count +=1
        for ele in candidate_.graph.getEdges():
            if ele.get_score() < 0:
                edge_count += 1
        data['best_{}'.format(idx)] = {'score': round(candidate_.graph.graph_score(), 2),
                                       'corner_score': round(candidate_.graph.corner_score(), 2),
                                       'edge_score': round(candidate_.graph.edge_score(), 2),
                                       'region_score': round(candidate_.graph.region_score(), 2),
                                       'false_corner': corner_count,
                                       'false_edge': edge_count}

    np.save(os.path.join(base_path, 'config'), data)

print('process training data')
train_dataset = trainSearchDataset(data_folder, data_scale=data_scale,
                                   edge_strong_constraint=edge_strong_constraint, corner_bin=False)
print('process testing data')
test_dataset = trainSearchDataset(data_folder, data_scale=data_scale,
                                  edge_strong_constraint=edge_strong_constraint, phase='valid',
                                  corner_bin=False)

search_dataset = myDataset(data_folder, phase='train', edge_linewidth=2, render_pad=-1)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

search_loader = torch.utils.data.DataLoader(search_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=1,
                                          drop_last=False)

# evaluator_train is used for training
# evaluator_search is used for searching
# separate into two modules in order to use multiple threads to accelerate
evaluator_train = scoreEvaluator_with_train('/local-scratch/fuyang/cities_dataset',
                                            backbone_channel=64, edge_bin_size=edge_bin_size,
                                            corner_bin=False)
#evaluator_train.load_weight(pretrained_path, 10)

evaluator_train.to('cuda:0')
evaluator_train.train()

optimizer = torch.optim.Adam(evaluator_train.parameters(), lr=1e-4)
cornerloss = nn.L1Loss()
heatmaploss = nn.MSELoss()
edgeloss = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1., 3.], device=evaluator_train.device))

os.makedirs(save_path, exist_ok=True)
f = open(os.path.join(save_path, 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')
######## start training and searching threads #####
lock = threading.Lock()
data_memory = []

st = searchThread(lock, None, search_loader, data_memory, train_dataset, search_dataset)
tt = trainThread(lock, evaluator_train, None, data_memory, train_loader, train_dataset, test_loader)

if activate_search_thread:
    st.start()
tt.start()

if activate_search_thread:
    st.join()
tt.join()

