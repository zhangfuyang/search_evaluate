import random
import numpy as np
from new_utils import visualization, Metric, Candidate, Graph, candidate_enumerate, reduce_duplicate_candidate
from colorama import Fore, Style
import torch
import torch.nn as nn
from new_config import *


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



def train(dataloader, old_model, model, edge_bin_size, optimizer, loss_func):
    model.train()

    heatmaploss = loss_func['heatmaploss']
    cornerloss = loss_func['cornerloss']
    edgeloss = loss_func['edgeloss']
    edge_psudo_loss = loss_func['edge_psudo_loss']

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
            loss_dict['edge_cross'] = edge_psudo_loss(edge_pred, edge_gt_pseudo.unsqueeze(1) + gamma * next_edge_pred.detach())

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
