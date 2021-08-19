import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import pickle
from config import config
import cv2
import skimage
import torch
from dataset import myDataset
from scoreAgent import scoreEvaluator_with_train as Model
from SVG_utils import svg_generate
from new_utils import visualization, candidate_enumerate, reduce_duplicate_candidate, Graph, Candidate

data_folder = config['data_folder']
beam_width = 5
beam_depth = 12
is_visualize = False
is_save = True
save_path = config['save_path']
phase = 'test'
prefix = '5'
use_smc = False
save_name = '{}_prefix_{}_convmpn'.format(phase, prefix)
use_heat_map = True 
print(save_path, save_name)

search_dataset = myDataset(data_folder, phase=phase)

search_loader = torch.utils.data.DataLoader(search_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            drop_last=False)

device = torch.device("cuda:0")
evaluator_search = Model(config['data_folder'], device, backbone_channel=64)
evaluator_search.load_weight(config['save_path'], prefix)
path = './data/pretrained_heatmap'
with open(os.path.join(path, '{}_{}.pt'.format(9, 'heatmapNet')), 'rb') as f:
    evaluator_search.heatmapNet.load_state_dict(torch.load(f))
evaluator_search.to(device)
evaluator_search.eval()


def search(evaluator):
    for idx, data in enumerate(search_loader):
        name = data['name'][0]
        print(name)
        if os.path.exists(os.path.join(save_path, save_name, name)):
            continue
       
        graph_data = search_dataset.getDataByName(name)
        conv_data = graph_data['conv_data']
        corners = conv_data['corners']
        corners = np.round(corners).astype(int)
        edges = conv_data['edges']

        gt_data = graph_data['gt_data']
        gt_corners = gt_data['corners']
        gt_corners = np.round(gt_corners).astype(int)
        gt_edges = gt_data['edges']
        
        # pre-compute heatmap and image feature (fast version)
        img = evaluator.img_cache.get_image(name)
        img = img.transpose((2,0,1))
        img = (img - np.array(config['mean'])[:, np.newaxis, np.newaxis]) / np.array(config['std'])[:, np.newaxis, np.newaxis]
        img = torch.cuda.FloatTensor(img, device=device).unsqueeze(0)
        with torch.no_grad():
            img_volume = evaluator.imgvolume(img)
            heatmap = evaluator.getheatmap(img)
        
        # gt
        gt_candidate = Candidate.initial(Graph(gt_corners, gt_edges), name)
        evaluator.get_score(gt_candidate, img_volume=img_volume, heatmap=heatmap)

        initial_candidate = Candidate.initial(Graph(corners, edges), name)
        evaluator.get_score(initial_candidate, img_volume=img_volume, heatmap=heatmap)

        # candidate gallery
        candidate_gallery = []
        candidate_gallery.append([initial_candidate])

        prev_candidates = [initial_candidate]
        best_candidates = [initial_candidate]
        best_count = 0

        for epoch_i in range(beam_depth):
            current_candidates = []
            for prev_i in range(len(prev_candidates)):
                prev_ = prev_candidates[prev_i]
                current_candidates.extend(candidate_enumerate(prev_))

            current_candidates = reduce_duplicate_candidate(current_candidates)
            print(epoch_i, len(current_candidates))
            for candidate_ in current_candidates:
                evaluator.get_score(candidate_, img_volume=img_volume, heatmap=heatmap)

            for candidate_i in range(len(current_candidates)):
                if best_candidates[0].graph.graph_score() < current_candidates[candidate_i].graph.graph_score():
                    best_candidates = [current_candidates[candidate_i]]
                    best_count = 0
                elif best_candidates[0].graph.graph_score() == current_candidates[candidate_i].graph.graph_score():
                    best_candidates.append(current_candidates[candidate_i])
                    best_count = 0
            best_count += 1
            
            if True:
                current_candidates = sorted(current_candidates, key=lambda x: x.graph.graph_score(), reverse=True)
                
                if len(current_candidates) < beam_width:
                    pick = np.arange(len(current_candidates))
                else:
                    pick = np.arange(beam_width)

                prev_candidates = [current_candidates[_] for _ in pick]

            for candidate_ in prev_candidates:
                candidate_.update()  # update safe_count
            candidate_gallery.append(prev_candidates)

            if best_count == 4:
                break

        ################################ save heat map ###########################################
        if use_heat_map:
            img = skimage.img_as_float(plt.imread(os.path.join(data_folder, 'rgb', name + '.jpg')))
            img = img.transpose((2, 0, 1))
            img = (img - np.array(config['mean'])[:, np.newaxis, np.newaxis]) / np.array(config['std'])[:, np.newaxis, np.newaxis]
            img = torch.cuda.FloatTensor(img, device=evaluator_search.device).unsqueeze(0)
            with torch.no_grad():
                heatmap = evaluator_search.getheatmap(img)
            heatmap = heatmap[0].cpu().numpy()
            heatmap = np.concatenate((heatmap, np.zeros((1, 256, 256))), 0).transpose(1, 2, 0)
            fig = plt.figure(frameon=False)
            fig.set_size_inches(1, 1)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(heatmap, aspect='auto')
            os.makedirs(os.path.join(save_path, save_name, name), exist_ok=True)
            fig.savefig(os.path.join(save_path, save_name, name, 'heatmap.png'),
                        dpi=256)
            plt.close()

        save_gallery(candidate_gallery, name,
                     os.path.join(save_path, save_name),
                     best_candidates, gt_candidate)


def save_candidate_image(candidate, base_path, base_name):
    corners = candidate.graph.getCornersArray()
    edges = candidate.graph.getEdgesArray()
    # graph svg
    svg = svg_generate(corners, edges, base_name, samecolor=True)
    svg.saveas(os.path.join(base_path, base_name + '.svg'))
    # corner image
    temp_mask = np.zeros((256, 256))
    temp_mask[0:8, :] = np.arange(256) / 255
    for ele in candidate.graph.getCorners():
        temp_mask = cv2.circle(temp_mask, ele.x[::-1], 3, (-ele.get_score() + 1) / 2, -1)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name + '_corner.png'), dpi=256)
    # edges image
    temp_mask = np.zeros((256, 256))
    temp_mask[0:8, :] = np.arange(256) / 255
    for ele in candidate.graph.getEdges():
        A = ele.x[0]
        B = ele.x[1]
        temp_mask = cv2.line(temp_mask, A.x[::-1], B.x[::-1], (-ele.get_score() + 1) / 2, thickness=2)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name + '_edge.png'), dpi=256)
    # region no need fig
    plt.close()


def save_candidate(candidate, name):
    f = open(name, 'wb')
    pickle.dump(candidate, f)
    f.close()


def save_gallery(candidate_gallery, name, save_path, best_candidates, gt_candidate):
    os.makedirs(save_path, exist_ok=True)
    base_path = os.path.join(save_path, name)
    os.makedirs(base_path, exist_ok=True)

    ############################### Image & MaskRcnn ####################################
    img_path = os.path.join(data_folder, 'rgb', name + '.jpg')
    img = plt.imread(img_path)
    plt.imsave(os.path.join(base_path, 'image.jpg'), img)

    ##################################### GT ############################################
    base_name = 'gt_pred'
    save_candidate_image(gt_candidate, base_path, base_name)
    save_candidate(gt_candidate, os.path.join(base_path, base_name + '.obj'))

    ################################### search ##########################################
    for k in range(len(candidate_gallery)):
        current_candidates = candidate_gallery[k]
        for idx, candidate_ in enumerate(current_candidates):
            base_name = 'iter_' + str(k) + '_num_' + str(idx)
            save_candidate_image(candidate_, base_path, base_name)
            save_candidate(candidate_, os.path.join(base_path, base_name + '.obj'))

    #################################### best ###########################################
    for k in range(len(best_candidates)):
        candidate_ = best_candidates[k]
        base_name = 'best_' + str(k)
        save_candidate_image(candidate_, base_path, base_name)
        save_candidate(candidate_, os.path.join(base_path, base_name + '.obj'))

    ################################ save config ########################################
    data = {}
    # gt
    corner_count = 0
    edge_count = 0
    for ele in gt_candidate.graph.getCorners():
        if ele.get_score() < 0: 
            corner_count += 1
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
                    corner_count += 1
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
                corner_count += 1
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


search(evaluator_search)
