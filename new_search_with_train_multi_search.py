import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
from new_train_test_func import test, train
from new_config import *
import pdb

print(config)


class trainThread(threading.Thread):
    def __init__(self, lock, evaluator, search_evaluator1, search_evaluator2, prev_evaluator, new_data_memory, dataloader, dataset, testdataset):
        super(trainThread, self).__init__()
        self.lock = lock
        self.evaluator = evaluator
        self.search_evaluator1 = search_evaluator1
        self.search_evaluator2 = search_evaluator2
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
            while len(self.dataset) < 2500:
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
                self.prev_evaluator.load_state_dict(self.search_evaluator1.state_dict())
                self.lock.release()
                train(self.dataloader, self.prev_evaluator, self.evaluator, edge_bin_size, optimizer, loss_func)

                while len(self.new_data_memory) != 0:
                    data = self.new_data_memory.pop()
                    self.dataset.add_processed_data(data)

                # update search evaluator
                self.lock.acquire()
                self.search_evaluator1.load_state_dict(self.evaluator.state_dict())
                self.search_evaluator2.load_state_dict(self.evaluator.state_dict())
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
    def __init__(self, thread_id, lock, evaluator, new_data_memory, trainDataset, searchDataset):
        super(searchThread, self).__init__()
        self.lock = lock
        self.thread_id = thread_id
        self.evaluator = evaluator
        self.new_data_memory = new_data_memory
        self.train_dataset = trainDataset
        self.search_dataset = searchDataset

    def run(self):
        print('{}[searching thread {}]{} start'.format(Fore.RED, self.thread_id, Style.RESET_ALL))
        search_count = 0
        add_count = 0
        #buffer = []

        while True:
            order = list(range(len(self.search_dataset)))
            random.shuffle(order)
            for idx in order:
                data = self.search_dataset.database[idx]
                name = data['name']
                conv_data = data['conv_data']
                corners = conv_data['corners']
                corners = np.round(corners).astype(np.int)
                edges = conv_data['edges']

                gt_data = data['gt_data']
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
                    current_candidates = sorted(current_candidates, key=lambda x:x.graph.graph_score(), reverse=True)

                    if len(current_candidates) == 0:
                        break
                    if len(current_candidates) > beam_width:
                        if random.random() < 0.8:
                            prev_candidates = current_candidates[:beam_width]
                        else:
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
                    print('{}[searching thread {}]{} Already search {} graphs and add {} '
                          'graphs into database'.format(Fore.RED, self.thread_id, Style.RESET_ALL, search_count, add_count))
                    print('{}[searching thread {}]{} {} remain in the swap'
                          ' memory'.format(Fore.RED, self.thread_id, Style.RESET_ALL, len(self.new_data_memory)))


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

search_dataset1 = myDataset(data_folder, phase='train', edge_linewidth=2, render_pad=-1)
search_dataset2 = myDataset(data_folder, phase='train', edge_linewidth=2, render_pad=-1)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=False)

# evaluator_train is used for training
# evaluator_search is used for searching
# separate into two modules in order to use multiple threads to accelerate
evaluator_train = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                            backbone_channel=64, edge_bin_size=edge_bin_size,
                                            corner_bin=False)
evaluator_search1 = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                              backbone_channel=64, edge_bin_size=edge_bin_size,
                                              corner_bin=False)
evaluator_search1.load_state_dict(evaluator_train.state_dict())
evaluator_search1.to('cuda:1')
evaluator_search1.eval()

evaluator_search2 = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                              backbone_channel=64, edge_bin_size=edge_bin_size,
                                              corner_bin=False)
evaluator_search2.load_state_dict(evaluator_train.state_dict())
evaluator_search2.to('cuda:2')
evaluator_search2.eval()

prev_evaluator = scoreEvaluator_with_train(os.path.join(base_path, 'cities_dataset'),
                                           backbone_channel=64, edge_bin_size=edge_bin_size,
                                           corner_bin=False)
prev_evaluator.load_state_dict(evaluator_search1.state_dict())
prev_evaluator.to('cuda:3')
prev_evaluator.eval()


evaluator_train.to('cuda:0')
evaluator_train.train()

optimizer = torch.optim.Adam(evaluator_train.parameters(), lr=1e-4)
cornerloss = nn.SmoothL1Loss()
heatmaploss = nn.MSELoss()
edgeloss = nn.SmoothL1Loss()
edge_psudo_loss = nn.SmoothL1Loss()
#edge_psudo_loss = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1., 3.], device=evaluator_train.device))
loss_func = {'cornerloss': cornerloss, 'heatmaploss': heatmaploss,
             'edgeloss': edgeloss, 'edge_psudo_loss': edge_psudo_loss}


os.makedirs(save_path, exist_ok=True)
f = open(os.path.join(save_path, 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')
######## start training and searching threads #####
lock = threading.Lock()
data_memory = []

st1 = searchThread(1, lock, evaluator_search1, data_memory, train_dataset, search_dataset1)
st2 = searchThread(2, lock, evaluator_search2, data_memory, train_dataset, search_dataset2)

tt = trainThread(lock, evaluator_train, evaluator_search1, evaluator_search2, prev_evaluator, data_memory, train_loader, train_dataset, test_dataset)

# activate_search_thread:
st1.start()
st2.start()
tt.start()

