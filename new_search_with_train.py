import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import copy
import cv2
import torch
from new_dataset import myDataset, trainSearchDataset
from new_scoreAgent import scoreEvaluator_with_train
import torch.nn as nn
import threading
from SVG_utils import svg_generate
from new_utils import visualization, Metric, candidate_enumerate, candidate_enumerate_training, reduce_duplicate_candidate, Graph, Candidate
from colorama import Fore, Style
from dicttoxml import dicttoxml
from new_config import *
import pdb

print(config)

def test(dataset, model, edge_bin_size):
    model.eval()
    batch_count = 0
    order = list(range(len(dataset)))
    random.shuffle(order)

    width = 2
    depth = 6
    metric = Metric()

    # use edge score as metric
    total_edge_tp = 0
    total_edge_fp = 0
    total_edge_length = 0

    for idx in order:
        if batch_count % 3 == 0:
            print('{}[TEST]{} Already test {}'.format(Fore.GREEN, Style.RESET_ALL, batch_count))
        if batch_count > 10:  # only test 10
            break
        data = dataset.database[idx]
        name = data['name']
        conv_data = data['conv_data']
        corners = conv_data['corners']
        corners = np.round(corners).astype(np.int)
        edges = conv_data['edges']
        if edges.shape[0] > 25:
            continue

        gt_data = data['gt_data']

        initial_candidate = Candidate.initial(Graph(corners, edges), name)
        model.get_score(initial_candidate)

        prev_candidates = [initial_candidate]
        best_candidate = initial_candidate
        best_count = 0
        for epoch_i in range(depth):
            current_candidates = []
            for prev_i in range(len(prev_candidates)):
                prev_ = prev_candidates[prev_i]
                current_candidates.extend(candidate_enumerate(prev_))
            current_candidates = reduce_duplicate_candidate(current_candidates)

            model.get_score_list(current_candidates, all_edge=True)

            for candidate_i in range(len(current_candidates)):
                if best_candidate.graph.graph_score() < current_candidates[candidate_i].graph.graph_score():
                    best_candidate = current_candidates[candidate_i]
                    best_count = 0
            best_count += 1
            if best_count == 4:
                break

            current_candidates = sorted(current_candidates, key=lambda x: x.graph.graph_score(), reverse=True)
            if len(current_candidates) < width:
                pick = np.arange(len(current_candidates))
            else:
                pick = np.arange(width)

            prev_candidates = [current_candidates[_] for _ in pick]
            for candidate_ in prev_candidates:
                candidate_.update()

        best_corners = best_candidate.graph.getCornersArray()
        best_edges = best_candidate.graph.getEdgesArray()
        best_data = {'corners': best_corners, 'edges': best_edges}
        score = metric.calc(gt_data, best_data)

        total_edge_tp += score['edge_tp']
        total_edge_fp += score['edge_fp']
        total_edge_length += score['edge_length']

        batch_count += 1

    recall = total_edge_tp / (total_edge_length + 1e-8)
    precision = total_edge_tp / (total_edge_tp + total_edge_fp + 1e-8)
    f1 = 2.0 * precision * recall / (recall+precision+1e-8)

    print('{}[TEST]{} Edge f1 score is {}'.format(Fore.GREEN, Style.RESET_ALL, f1))

    return f1



def train(dataloader, old_model, model, edge_bin_size):
    model.train()
    total_loss = {}
    batch_count = 0
    for idx, data in enumerate(dataloader):
        img = data['img'].to(model.device)
        next_img = data['img'].to(old_model.device)
        mask = data['mask'].to(model.device)
        next_mask = data['next_mask'].to(old_model.device)
        corner_gt_mask = data['corner_gt_mask'].to(model.device)
        gt_edge = data['gt_edge'].to(model.device).squeeze(1)
        edge_mask = data['edge_mask'].to(model.device)
        next_edge_mask = data['next_edge_mask'].to(old_model.device)

        optimizer.zero_grad()

        loss_dict = {}

        img_volume = model.imgvolume(img)
        if use_heat_map:
            heatmap = model.getheatmap(img)
            heatmap_detach = heatmap
            gt_heat_map = data['gt_heat_map'].to(model.device)
            heatmap_l = heatmaploss(heatmap, gt_heat_map)
            loss_dict['heatmap'] = heatmap_l
        else:
            heatmap = None
            heatmap_detach = None
        corner_pred = model.cornerEvaluator(mask, img_volume, binmap=None, heatmap=heatmap_detach)
        edge_pred = model.edgeEvaluator(edge_mask, mask, img_volume, corner_pred.detach(),
                                        torch.zeros(edge_mask.shape[0], edge_bin_size, device=model.device),
                                        binmap=None, heatmap=heatmap_detach)


        with torch.no_grad():
            next_img_volume = old_model.imgvolume(next_img)
            if use_heat_map:
                next_heatmap = old_model.getheatmap(next_img)
                next_heatmap_detach = next_heatmap
            else:
                next_heatmap = None
                next_heatmap_detach = None

            next_corner_pred = old_model.cornerEvaluator(next_mask, next_img_volume, binmap=None, heatmap=next_heatmap_detach)
            next_edge_pred = old_model.edgeEvaluator(next_edge_mask, next_mask, next_img_volume, next_corner_pred.detach(),
                                    torch.zeros(edge_mask.shape[0], edge_bin_size, device=old_model.device),
                                    binmap=None, heatmap=next_heatmap_detach).detach().cpu()

        next_edge_pred = next_edge_pred.to(model.device)
        next_corner_pred = next_corner_pred.to(model.device)

        # discounted corner loss
        corner_l = cornerloss(corner_pred, corner_gt_mask + gamma * next_corner_pred.detach())  # (bs, 1, 256, 256)  after sigmoid

        # discounted edge loss
        edge_l = edgeloss(edge_pred, gt_edge.unsqueeze(1) + gamma * next_edge_pred.detach())

        loss_dict['corner'] = corner_l
        loss_dict['edge'] = edge_l

        '''
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
            loss_dict['edge_cross'] = edge_psudo_loss(edge_pred, edge_gt_pseudo)

            # corner pseudo
            ### test ###
            #corner_input_mask = (mask[:, 1:]+1)/2
            #corner_gt_pseudo = torch.mul(corner_input_mask, pseudo_corner_map)
            #loss_dict['corner_cross'] = cornerloss(corner_pred, corner_gt_pseudo)
        '''
        loss = 0
        for key in loss_dict.keys():
            if key == 'corner':
                loss += loss_dict[key] * 10
            elif key == 'edge':
                loss += loss_dict[key]
            elif key == 'heatmap':
                loss += loss_dict[key] * 5
            elif key == 'edge_cross':
                loss += loss_dict[key] * 0.1
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
    def __init__(self, lock, evaluator, search_evaluator, prev_evaluator, new_data_memory, dataloader, dataset, testdataset):
        super(trainThread, self).__init__()
        self.lock = lock
        self.evaluator = evaluator
        self.search_evaluator = search_evaluator
        self.new_data_memory = new_data_memory
        self.dataloader = dataloader
        self.dataset = dataset
        self.testdataset = testdataset
        self.prev_evaluator = prev_evaluator

    def run(self):
        print('{}[training thread]{} wait for initial data'.format(Fore.BLUE, Style.RESET_ALL))
        prefix = 0
        testbest = 0
        test_f1 = test(self.testdataset, self.evaluator, edge_bin_size)
        while True:
            # pre-load data
            while len(self.dataset) < 1500:
                time.sleep(30)
                while len(self.new_data_memory) != 0:
                    data = self.new_data_memory.pop()
                    self.dataset.add_processed_data(data)
                continue

            print('{}[training thread]{} start training'.format(
                    Fore.BLUE, Style.RESET_ALL))
            train_sample = 0
            for _ in range(3):
                print('{}[training thread]{} New start {}! training with {} samples'.format(
                    Fore.BLUE, Style.RESET_ALL, prefix, len(self.dataset)))
                self.lock.acquire()
                self.prev_evaluator.load_state_dict(self.search_evaluator.state_dict())
                self.lock.release()
                train(self.dataloader, self.prev_evaluator, self.evaluator, edge_bin_size)
                
                while len(self.new_data_memory) != 0:
                    data = self.new_data_memory.pop()
                    self.dataset.add_processed_data(data)

                # update search evaluator
                self.lock.acquire()
                self.search_evaluator.load_state_dict(self.evaluator.state_dict())
                self.lock.release()

                train_sample += len(self.dataset)
                if train_sample >= MAX_DATA_STORAGE/2:
                    break

            test_f1 = test(self.testdataset, self.evaluator, edge_bin_size)
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
            if test_f1 > testbest:
                testbest = test_f1
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
        #buffer = []

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
                        next_candidates = candidate_enumerate_training(prev_, gt_data)

                        self.evaluator.get_score_list(next_candidates, all_edge=True)
                        next_candidates = sorted(next_candidates, key=lambda x:x.graph.graph_score(), reverse=True)
                        # pick the best as S_{t+1}
                        next_ = next_candidates[0]

                        prev_corners = prev_.graph.getCornersArray()
                        prev_edges = prev_.graph.getEdgesArray()
                        next_corners = next_.graph.getCornersArray()
                        next_edges = next_.graph.getEdgesArray()
                        data = self.train_dataset.make_data(prev_.name, prev_corners, prev_edges, next_corners, next_edges)

                        self.new_data_memory.append(data)
                        add_count += 1

                        current_candidates.extend(next_candidates)

                    current_candidates = reduce_duplicate_candidate(current_candidates)

                    if len(current_candidates) == 0:
                        break
                    if len(current_candidates) > beam_width:
                        prev_candidates = random.sample(current_candidates, beam_width)
                    else:
                        prev_candidates = current_candidates

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
test_dataset = myDataset(data_folder, phase='valid', edge_linewidth=2, render_pad=-1)

search_dataset = myDataset(data_folder, phase='train', edge_linewidth=2, render_pad=-1)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)

search_loader = torch.utils.data.DataLoader(search_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=0,
                                          drop_last=False)

# evaluator_train is used for training
# evaluator_search is used for searching
# separate into two modules in order to use multiple threads to accelerate
evaluator_train = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                            backbone_channel=64, edge_bin_size=edge_bin_size,
                                            corner_bin=False)
evaluator_search = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                            backbone_channel=64, edge_bin_size=edge_bin_size,
                                            corner_bin=False)
evaluator_search.load_state_dict(evaluator_train.state_dict())
evaluator_search.to('cuda:1')
evaluator_search.eval()

prev_evaluator = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                             backbone_channel=64, edge_bin_size=edge_bin_size,
                                             corner_bin=False)
prev_evaluator.load_state_dict(evaluator_search.state_dict())
prev_evaluator.to('cuda:1')
prev_evaluator.eval()


evaluator_train.to('cuda:0')
evaluator_train.train()

optimizer = torch.optim.Adam(evaluator_train.parameters(), lr=1e-4)
cornerloss = nn.SmoothL1Loss()
heatmaploss = nn.MSELoss()
edgeloss = nn.SmoothL1Loss()
edge_psudo_loss = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1., 3.], device=evaluator_train.device))


os.makedirs(save_path, exist_ok=True)
f = open(os.path.join(save_path, 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')
######## start training and searching threads #####
lock = threading.Lock()
data_memory = []

st1 = searchThread(lock, evaluator_search, search_loader, data_memory, train_dataset, search_dataset)

tt = trainThread(lock, evaluator_train, evaluator_search, prev_evaluator, data_memory, train_loader, train_dataset, test_dataset)

# activate_search_thread:
st1.start()
tt.start()

