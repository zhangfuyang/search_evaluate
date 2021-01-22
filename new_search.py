import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch
from new_dataset import myDataset
from new_scoreAgent import scoreEvaluator
from tqdm import tqdm
import threading
from new_utils import visualization, candidate_enumerate, reduce_duplicate_candidate, Graph, Candidate
from SVG_utils import svg_generate

data_folder = '/local-scratch/fuyang/cities_dataset'
beam_width = 6
beam_depth = 10
is_visualize = False
is_save = True
save_path = '/local-scratch/fuyang/result/beam_search_v2/without_search_constraint/old/'


test_dataset = myDataset(data_folder, phase='valid', edge_linewidth=2, render_pad=-1)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=1,
                                          drop_last=True)

evaluator = scoreEvaluator({'cornerModelPath':'/local-scratch/fuyang/result/corner_v2/gt_mask_with_gt/models/440.pt',
                            'edgeModelPath':'/local-scratch/fuyang/result/corner_edge_region/edge_graph_drn26_with_search_v2/models/best.pt',
                            'regionModelPath': '/local-scratch/fuyang/result/corner_edge_region/region_graph_iter_0/models/70.pt',
                            'region_iter': 0,
                            'edgeHeatmapPath': '/local-scratch/fuyang/result/corner_edge_region/edge_heatmap_unet/all_edge_masks',
                            'regionHeatmapPath': '/local-scratch/fuyang/result/corner_edge_region/all_region_masks',
                            'regionEntireMaskPath': '/local-scratch/fuyang/result/corner_edge_region/entire_region_mask'
                            }, useHeatmap=('region'), useGT=(), dataset=test_dataset)


class _thread(threading.Thread):
    def __init__(self, threadID, name, candidate, lock, result_list, func):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.candidate = candidate
        self.lock = lock
        self.result_list = result_list
        self.func = func
    def run(self):
        print('running id: ', self.name)
        start_time = time.time()
        candidates = self.func(self.candidate)
        print('test: =================================', self.name, len(candidates))
        self.lock.acquire()
        self.result_list.extend(candidates)
        self.lock.release()
        print(self.name, "spend time: {}s".format(time.time()-start_time))



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


for idx, data in enumerate(test_loader):
    name = data['name'][0]
    #if name != '1553901866.54':
    #    continue
    img = data['img'][0]
    graph_data = test_dataset.getDataByName(name)
    conv_data = graph_data['conv_data']
    corners = conv_data['corners']
    corners = np.round(corners).astype(np.int)
    edges = conv_data['edges']

    gt_data = graph_data['gt_data']
    gt_corners = gt_data['corners']
    gt_corners = np.round(gt_corners).astype(np.int)
    gt_edges = gt_data['edges']

    # gt score
    gt_candidate = Candidate.initial(Graph(gt_corners, gt_edges), name)
    evaluator.get_score(gt_candidate)

    # initial score
    initial_candidate = Candidate.initial(Graph(corners, edges), name)

    print("####################################################################")
    print("####################################################################")
    print(name)
    print("####################################################################")
    print("####################################################################")
    evaluator.get_score(initial_candidate)

    # candidate gallery
    candidate_gallery = []
    candidate_gallery.append([initial_candidate])

    epoch = min(beam_depth, corners.shape[0]*2)
    prev_candidates = [initial_candidate]
    best_candidates = [initial_candidate]
    _best_count = 0
    for epoch_i in range(epoch):
        print("======================== epoch ", epoch_i, " =======================")
        start_time = time.time()
        current_candidates = []
        for prev_i in range(len(prev_candidates)):
            prev_ = prev_candidates[prev_i]
            current_candidates.extend(candidate_enumerate(prev_))

        print("all prev candidate enumerate done.", "overall", len(current_candidates), "candidates")

        current_candidates = reduce_duplicate_candidate(current_candidates)

        print("reduce duplicate candidates done.", "overall", len(current_candidates), "candidates")

        # uncomment to get robust fast score but slower
        #for candidate_i in tqdm(range(len(current_candidates))):
        #    evaluator.get_fast_score(current_candidates[candidate_i])

        eval_time = time.time()
        evaluator.get_fast_score_list(current_candidates)
        print('average time: {}s/sample'.format((time.time()-eval_time) / len(current_candidates)))

        for candidate_i in range(len(current_candidates)):
            if best_candidates[0].graph.graph_score() < current_candidates[candidate_i].graph.graph_score():
                best_candidates = [current_candidates[candidate_i]]
                _best_count = -1
            elif best_candidates[0].graph.graph_score() == current_candidates[candidate_i].graph.graph_score():
                best_candidates.append(current_candidates[candidate_i])
                _best_count = -1
        _best_count += 1


        print("finish evaluating all candidates")

        current_candidates = sorted(current_candidates, key=lambda x:x.graph.graph_score(), reverse=True)
        if len(current_candidates) < beam_width:
            pick = np.arange(len(current_candidates))
        else:
            pick = np.arange(beam_width)

        prev_candidates = [current_candidates[_] for _ in pick]

        # update safe_count
        for candidate_ in prev_candidates:
            candidate_.update()
        candidate_gallery.append(prev_candidates)
        print("========================finish epoch: ", epoch_i, "=====================================")
        print("spend time: {}s".format(time.time()-start_time))
        if _best_count == 3:
            break


    if is_save:
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





