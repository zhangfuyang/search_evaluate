import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    total_edge_loss = 0
    total_corner_loss = 0
    batch_count = 0
    for idx, data in enumerate(dataloader):
        img = data['img'].to(model.device)
        mask = data['mask'].to(model.device)
        corner_gt_mask = data['corner_gt_mask'].to(model.device)
        gt_edge = data['gt_edge'].to(model.device).squeeze(1)
        edge_mask = data['edge_mask'].to(model.device)

        with torch.no_grad():
            img_volume = model.imgvolume(img)
            corner_pred = model.cornerEvaluator(mask, img_volume)
            edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                        torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device))
            total_corner_loss += cornerloss(corner_pred, corner_gt_mask).item()
            total_edge_loss += edgeloss(edge_pred, gt_edge).item()
            batch_count += 1

    print('{}[TEST]{} corner_loss={} edge_loss={}'.format(Fore.GREEN, Style.RESET_ALL,
                                                          round(total_corner_loss/batch_count, 2),
                                                          round(total_edge_loss/batch_count, 2)))

def train(dataloader, model, edge_bin_size):
    model.train()
    total_edge_loss = 0
    total_corner_loss = 0
    batch_count = 0
    for idx, data in enumerate(dataloader):
        img = data['img'].to(model.device)
        mask = data['mask'].to(model.device)
        corner_gt_mask = data['corner_gt_mask'].to(model.device)
        gt_edge = data['gt_edge'].to(model.device).squeeze(1)
        edge_mask = data['edge_mask'].to(model.device)

        optimizer.zero_grad()

        img_volume = model.imgvolume(img)
        if use_corner_bin_map:
            binmap = model.getbinmap(img)
            corner_pred = model.cornerEvaluator(mask, img_volume, binmap.detach())
            edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                            torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device),
                                            binmap.detach())
        else:
            corner_pred = model.cornerEvaluator(mask, img_volume)
            edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                            torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device))
        loss1 = cornerloss(corner_pred, corner_gt_mask)
        loss2 = edgeloss(edge_pred, gt_edge)
        if use_cross_loss:
            pass #TODO
        loss = 30*loss1 + loss2
        if idx % 30 == 0:
            print('[Batch {}/{}] Corner Loss={}; Edge Loss={}'.format(idx, len(dataloader), loss1.item(), loss2.item()))
        total_corner_loss += loss1.item()
        total_edge_loss += loss2.item()
        batch_count += 1

        loss.backward()
        optimizer.step()

    print('[Overall train] corner_loss={} edge_loss={}'.format(round(total_corner_loss/batch_count, 2),
                                                          round(total_edge_loss/batch_count, 2)))

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
        while True:
            train_sample = 0
            for _ in range(4):
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

            test(self.testloader, self.evaluator, edge_bin_size)
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

class CrossLoss(nn.Module):
    def __init__(self):
        super(CrossLoss, self).__init__()
    def forward(self, corner_pred, ):
        pass
    #TODO edge segmentation and corner dir


print('process training data')
train_dataset = trainSearchDataset(data_folder, data_scale=data_scale,
                                   edge_strong_constraint=edge_strong_constraint, corner_bin=use_corner_bin_map)
print('process testing data')
test_dataset = trainSearchDataset(data_folder, data_scale=data_scale,
                                  edge_strong_constraint=edge_strong_constraint, phase='valid',
                                  corner_bin=use_corner_bin_map)

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
                                            corner_bin=use_corner_bin_map)
#evaluator_train.load_weight(pretrained_path, 10)

evaluator_train.to('cuda:0')
evaluator_train.train()

optimizer = torch.optim.Adam(evaluator_train.parameters(), lr=1e-4)
cornerloss = nn.L1Loss()
edgeloss = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1., 3.], device=evaluator_train.device))
if use_cross_loss:
    crossloss = CrossLoss()

os.makedirs(save_path, exist_ok=True)
f = open(os.path.join(save_path, 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
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

