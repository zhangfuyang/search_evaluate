import torch
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from model import cornerModel, region_model
from drn import drn_c_26
from utils import *

class scoreEvaluator():
    def __init__(self, modelPath, useHeatmap=(), useGT=(), dataset=None, continuous_score=False):
        self.useHeatmap = useHeatmap
        self.useGT = useGT
        self.dataset = dataset
        self.continuous_score = continuous_score
        if 'cornerModelPath' in modelPath.keys():
            self.cornermodelPath = modelPath['cornerModelPath']
            self.cornerEvaluator = cornerModel()
            with open(self.cornermodelPath, 'rb') as f:
                state_dict = torch.load(f)
                self.cornerEvaluator.load_state_dict(state_dict)
                self.cornerEvaluator.double()
                self.cornerEvaluator.cuda()
                self.cornerEvaluator.eval()
        else:
            self.cornerEvaluator = None

        if 'edge' in self.useHeatmap:
            self.edge_use_heatmap = True
            self.edgeHeatmapPath = modelPath['edgeHeatmapPath']
        else:
            self.edge_use_heatmap = False

        if 'region' in self.useHeatmap:
            self.region_use_heatmap = True
            #self.regionHeatmapPath = modelPath['regionHeatmapPath']
            self.regionHeatmapPath = modelPath['regionEntireMaskPath']
        else:
            self.region_use_heatmap = False

        if 'corner' in self.useGT:
            self.corner_use_gt = True
        else:
            self.corner_use_gt = False
        if 'edge' in self.useGT:
            self.edge_use_gt = True
        else:
            self.edge_use_gt = False
        if 'region' in self.useGT:
            self.region_use_gt = True
        else:
            self.region_use_gt = False

        if 'edgeModelPath' in modelPath.keys():
            self.edgemodelPath = modelPath['edgeModelPath']
            self.edgeEvaluator = drn_c_26(pretrained=True, num_classes=2, in_channels=6)
            with open(self.edgemodelPath, 'rb') as f:
                state_dict = torch.load(f)
                self.edgeEvaluator.load_state_dict(state_dict, resnet_pretrained=False)
                self.edgeEvaluator.double()
                self.edgeEvaluator.cuda()
                self.edgeEvaluator.eval()
        else:
            self.edgeEvaluator = None

        if 'regionModelPath' in modelPath.keys():
            self.regionmodelPath = modelPath['regionModelPath']
            self.regionEvaluator = region_model(iters=modelPath['region_iter'])
            with open(self.regionmodelPath, 'rb') as f:
                state_dict = torch.load(f)
                self.regionEvaluator.load_state_dict(state_dict)
                self.regionEvaluator.double()
                self.regionEvaluator.cuda()
                self.regionEvaluator.eval()
        else:
            self.regionEvaluator = None

    def corner_map2score(self, corners, map):
        corner_state = np.ones(corners.shape[0])
        for corner_i in range(corners.shape[0]):
            loc = np.round(corners[corner_i]).astype(np.int)
            if loc[0] <= 1:
                x0 = 0
            else:
                x0 = loc[0] - 1
            if loc[0] >= 254:
                x1 = 256
            else:
                x1 = loc[0] + 2

            if loc[1] <= 1:
                y0 = 0
            else:
                y0 = loc[1] - 1
            if loc[1] >= 254:
                y1 = 256
            else:
                y1 = loc[1] + 2
            heat = map[x0:x1, y0:y1]
            corner_state[corner_i] = 1-heat.sum()/heat.shape[0]/heat.shape[1] * 2  #[-1, 1]
        return corner_state

    def get_list_candidates_score(self, img, graphs, name=None):
        scores = []
        configs = []
        for graph_i in range(len(graphs)):
            score, score_config = self.get_score(img=img, corners=graphs[graph_i][0], edges=graphs[graph_i][1], name=name)
            scores.append(score)
            configs.append(score_config)
        #N = 16
        #scores = np.array([])
        #group = np.ceil(len(graphs)/N).astype(np.int)
        #for group_i in range(group):
        #    if group_i == group-1:
        #        gs = graphs[N*group_i:]
        #    else:
        #        gs = graphs[N*group_i:N*group_i+N]
        #    masks = []
        #    for i in range(len(gs)):
        #        corners = gs[i][0]
        #        edges = gs[i][1]
        #        mask = render(corners, edges, -1, 2)
        #        masks.append(mask)
        #    masks = torch.Tensor(masks).double()
        #    corners_group = [gs[i][0] for i in range(len(gs))]
        #    scores_group = self.get_score(img, masks, corners_group)
        #    scores = np.concatenate((scores, scores_group), 0)

        return scores, configs

    def get_score(self, img, corners, edges, name=None):
        """
        :param img:  3x256x256
        :param edges: Ex2
        :param corners: Vx2
        :return: N numpy
        """
        graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
        graph_mask = torch.Tensor(graph_mask).double()
        graph_mask = graph_mask.cuda()
        img = img.cuda()
        img = img.unsqueeze(0)

        score = {}

        if self.corner_use_gt or self.edge_use_gt or self.region_use_gt:
            gt_data = self.dataset.getDataByName(name)['gt_data']

        #######################   corner   ###############################
        if self.corner_use_gt:
            false_corners, gt_match, conv2gt_map, _ = get_wrong_corners(corners=corners, gt_corners=gt_data['corners'],
                                                                    edges=edges, gt_edges=gt_data['edges'])

            false_corner_id = list(false_corners)
            score['corner'] = corners.shape[0] - 2*len(false_corner_id)
            corners_score = []
            for corner_i in range(corners.shape[0]):
                if corner_i in false_corner_id:
                    corners_score.append(-1.)
                else:
                    corners_score.append(1.)
            score['corners_state'] = corners_score
        elif self.cornerEvaluator is not None:
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat((img, graph_mask), 1))
            corner_map = corner_map.cpu().numpy()[0,0]
            corners_state = self.corner_map2score(corners, corner_map)
            if self.continuous_score:
                score['corner'] = corners_state.sum()
            else:
                score['corner'] = (corners_state >= 0).sum() - (corners_state < 0).sum()
            score['corners_state'] = list(corners_state)

        #######################   edge     ###############################
        if self.edge_use_gt:
            false_corners, gt_match, conv2gt_map, _ = get_wrong_corners(corners=corners, gt_corners=gt_data['corners'],
                                                                    edges=edges, gt_edges=gt_data['edges'])

            false_edge_id = []
            for edge_i in range(edges.shape[0]):
                id1 = edges[edge_i][0]
                id2 = edges[edge_i][1]
                if id1 not in conv2gt_map.keys() or id2 not in conv2gt_map.keys():
                    false_edge_id.append(edge_i)
                    continue
                gt_id1 = conv2gt_map[id1]
                gt_id2 = conv2gt_map[id2]
                flag = False
                for edge_j in range(gt_data['edges'].shape[0]):
                    if gt_data['edges'][edge_j][0] == gt_id1 and gt_data['edges'][edge_j][1] == gt_id2:
                        flag = True
                        break
                    if gt_data['edges'][edge_j][0] == gt_id2 and gt_data['edges'][edge_j][1] == gt_id1:
                        flag = True
                        break
                if flag == False:
                    false_edge_id.append(edge_i)
            score['edge'] = edges.shape[0] - 2*len(false_edge_id)
            edges_state = []
            for edge_i in range(edges.shape[0]):
                if edge_i in false_edge_id:
                    edges_state.append(-1.)
                else:
                    edges_state.append(1.)
            score['edges_state'] = edges_state
        elif self.edge_use_heatmap:
            pred_edge_mask = np.load(os.path.join(self.edgeHeatmapPath, name+'.npy'), allow_pickle=True)[0][0]
            edges_state = []
            for edge_i in range(edges.shape[0]):
                corner1 = np.round(corners[edges[edge_i,0]])
                corner2 = np.round(corners[edges[edge_i,1]])
                temp_mask = np.zeros((256,256))
                temp_mask = cv2.line(temp_mask, (int(corner1[1]), int(corner1[0])),
                                     (int(corner2[1]), int(corner2[0])), 1.0, thickness=1)
                temp_mask2 = np.zeros((256,256))
                temp_mask2 = cv2.line(temp_mask2, (int(corner1[1]), int(corner1[0])),
                                      (int(corner2[1]), int(corner2[0])), 1.0, thickness=2)
                temp_score = np.logical_and(pred_edge_mask > 0.5, temp_mask2).sum() / temp_mask.sum()
                edges_state.append(temp_score*2-1)  # 0.5 is middle #TODO can change
            if self.continuous_score:
                score['edge'] = sum(edges_state)
            else:
                score['edge'] = (np.array(edges_state) >= 0).sum() - (np.array(edges_state) < 0).sum()
            score['edges_state'] = edges_state
        else:
            if self.edgeEvaluator is not None:
                batchs = patch_samples(edges.shape[0], 16)

                edge_score = np.array([])
                for batch in batchs:
                    temp_masks = []
                    for edge_i in batch:
                        a = edges[edge_i,0]
                        b = edges[edge_i,1]
                        temp_mask = cv2.line(np.ones((256,256))*-1, (int(corners[a,1]), int(corners[a,0])),
                                             (int(corners[b,1]), int(corners[b,0])),
                                             1.0, thickness=2)[np.newaxis, ...]
                        temp_mask = torch.Tensor(temp_mask).unsqueeze(0).double()
                        temp_masks.append(temp_mask)
                    temp_masks = torch.cat(temp_masks, 0)
                    with torch.no_grad():
                        edge_masks = temp_masks.cuda()
                        #####
                        graph_mask_ex = graph_mask.expand(edge_masks.shape[0], -1, -1, -1)
                        images = img.expand(edge_masks.shape[0], -1, -1, -1)
                        out = self.edgeEvaluator(torch.cat((images, graph_mask_ex, edge_masks), 1))
                    out = out.cpu()
                    edge_batch_score = out.exp()[:,0] / out.exp().sum(1)
                    edge_batch_score = edge_batch_score.numpy()
                    edge_score = np.append(edge_score, edge_batch_score)
                edges_state = []
                for edge_i in range(edge_score.shape[0]):
                    edges_state.append(-1.9*edge_score[edge_i]*edge_score[edge_i]-0.1*edge_score[edge_i]+1)
            if self.continuous_score:
                score['edge'] = sum(edges_state)
            else:
                score['edge'] = (np.array(edges_state) >= 0).sum() - (np.array(edges_state) < 0).sum()
            score['edges_state'] = edges_state

        #######################   region   ###############################
        if self.region_use_gt:
            ## produce gt masks
            gt_edge_mask = render(corners=gt_data['corners'], edges=gt_data['edges'], render_pad=0, edge_linewidth=1)[0]
            gt_edge_mask = 1 - gt_edge_mask
            gt_edge_mask = gt_edge_mask.astype(np.uint8)
            labels, gt_region_mask = cv2.connectedComponents(gt_edge_mask, connectivity=4)
            background_label = gt_region_mask[0,0]
            gt_masks = []
            for region_i in range(1, labels):
                if region_i == background_label:
                    continue
                curr_region = gt_region_mask == region_i
                if curr_region.sum() < 20:
                    continue
                gt_masks.append(curr_region)

            edge_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
            edge_mask = 1 - edge_mask
            edge_mask = edge_mask.astype(np.uint8)
            labels, region_mask = cv2.connectedComponents(edge_mask, connectivity=4)
            background_label = region_mask[0,0]
            false_region_id = []
            region_num = 0
            pred_region_masks = []
            for region_i in range(1, labels):
                if region_i == background_label:
                    continue
                curr_region = region_mask == region_i
                if curr_region.sum() < 20:
                    continue
                region_num += 1
                pred_region_masks.append((curr_region, region_i))

            temp_used_pred = set()
            temp_gt_map = []
            pred_map_gt = [-1 for _ in range(len(pred_region_masks))]
            for gt_i in range(len(gt_masks)):
                best_iou = 0.5
                best_pred_idx = -1
                for pred_i in range(len(pred_region_masks)):
                    if pred_i in temp_used_pred:
                        continue
                    iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_i
                if best_pred_idx != -1:
                    temp_used_pred.add(best_pred_idx)
                    pred_map_gt[best_pred_idx] = gt_i
                temp_gt_map.append(best_pred_idx)
            for pred_i in range(len(pred_region_masks)):
                if pred_map_gt[pred_i] == -1:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_i in range(len(gt_masks)):
                        if temp_gt_map[gt_i] != -1:
                            continue
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_i
                    pred_map_gt[pred_i] = best_gt_idx

            regions_state = []
            regions_number = []
            for pred_i in range(len(pred_region_masks)):
                gt_idx = pred_map_gt[pred_i]
                if gt_idx == -1:
                    iou = 0
                else:
                    iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_idx])
                regions_state.append(iou*2-1)
                regions_number.append(pred_region_masks[pred_i][1])

            if self.continuous_score:
                score['region'] = sum(regions_state)
            else:
                score['region'] = (np.array(regions_state) >= 0).sum() - (np.array(regions_state) < 0).sum()
            score['regions_state'] = regions_state
            score['regions_id'] = regions_number
        elif self.region_use_heatmap:
            if 'entire' in self.regionHeatmapPath and self.continuous_score:
                gt_mask = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'))
                gt_mask = gt_mask > 0.4
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                all_masks = []
                regions_number = []
                for region_i in range(1, labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    all_masks.append(the_region)
                    regions_number.append(region_i)

                pred_mask = (np.sum(all_masks, 0) + (1 - conv_mask))>0

                iou = IOU(pred_mask, gt_mask)
                score['region'] = iou
                score['regions_state'] = [1 for _ in regions_number]
                score['regions_id'] = regions_number

            else:
                gt_masks = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'), allow_pickle=True)
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                false_region_id = []
                region_num = 0
                pred_region_masks = []
                for region_i in range(1,labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    region_num += 1
                    pred_region_masks.append((the_region, region_i))

                temp_used_pred = set()
                temp_gt_map = []
                pred_map_gt = [-1 for _ in range(len(pred_region_masks))]
                for gt_i in range(gt_masks.shape[0]):
                    best_iou = 0.5
                    best_pred_idx = -1
                    for pred_i in range(len(pred_region_masks)):
                        if pred_i in temp_used_pred:
                            continue
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_i
                    if best_pred_idx != -1:
                        temp_used_pred.add(best_pred_idx)
                        pred_map_gt[best_pred_idx] = gt_i
                    temp_gt_map.append(best_pred_idx)
                for pred_i in range(len(pred_region_masks)):
                    if pred_map_gt[pred_i] == -1:
                        best_iou = 0
                        best_gt_idx = -1
                        for gt_i in range(gt_masks.shape[0]):
                            if temp_gt_map[gt_i] != -1:
                                continue
                            iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_i
                        pred_map_gt[pred_i] = best_gt_idx

                regions_state = []
                regions_number = []
                for pred_i in range(len(pred_region_masks)):
                    gt_idx = pred_map_gt[pred_i]
                    if gt_idx == -1:
                        iou = 0
                    else:
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_idx])
                    regions_state.append(iou*2-1)
                    regions_number.append(pred_region_masks[pred_i][1])

                if self.continuous_score:
                    score['region'] = sum(regions_state)
                else:
                    score['region'] = (np.array(regions_state) >= 0).sum() - (np.array(regions_state) < 0).sum()
                score['regions_state'] = regions_state
                score['regions_id'] = regions_number
        else:
            if self.regionEvaluator is not None:
                input_edge_mask = render(corners, edges, -1,2)
                temp_edge_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                temp_edge_mask = 1 - temp_edge_mask
                temp_edge_mask = temp_edge_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(temp_edge_mask, connectivity=4)
                background_label = region_mask[0,0]
                false_region_id = []
                input_masks = []
                region_num = 0
                region_id = []
                regions_number = []
                for region_i in range(1, labels):
                    if region_i == background_label:
                        continue
                    curr_region = region_mask == region_i
                    if curr_region.sum() < 20:
                        continue
                    region_num += 1
                    region_id.append(region_i)
                    input_masks.append(curr_region)
                    regions_number.append(region_i)
                input_masks = torch.Tensor(input_masks).double().unsqueeze(1) * 2 - 1
                input_masks = input_masks.cuda()
                input_edge_mask = torch.Tensor(input_edge_mask).double().cuda().unsqueeze(0)
                if input_masks.shape[0] == 0:
                    region_num = 0
                    regions_state = []
                else:
                    with torch.no_grad():
                        region_pred = self.regionEvaluator(img, input_edge_mask, input_masks)
                    region_score = region_pred.exp()[:,1] / region_pred.exp().sum(1)
                    region_score = region_score.cpu().numpy()
                    regions_state = list(region_score * 2 - 1)

            if self.continuous_score:
                score['region'] = sum(regions_state)
            else:
                score['region'] = (np.array(regions_state) >= 0).sum() - (np.array(regions_state) < 0).sum()
            score['regions_state'] = regions_state
            score['regions_id'] = regions_number

        #plt.imshow(np.concatenate((np.concatenate((mask, map), 0)[[0,2]].transpose(1,2,0), np.zeros((256,256,1))),2))
        #plt.show()
        score_all = score['corner'] + 2*score['edge'] + 60*score['region']

        #### debug ####
        #img = skimage.img_as_float(plt.imread(os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name+'.jpg')))
        #plt.subplot(241)
        #plt.title(name)
        #plt.imshow(img)
        #plt.subplot(242)
        #plt.title('conv-mpn')
        #conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)
        #conv_mask = conv_mask[0]+conv_mask[1]
        #plt.imshow(conv_mask)
        #plt.subplot(243)
        #plt.title('all regions:' + str(region_num))
        #plt.imshow(region_mask)
        #plt.subplot(244)
        #plt.title('edge heat map')
        #plt.imshow(pred_edge_mask)
        #plt.subplot(245)
        #plt.title('false corner: '+str(len(false_corner_id)))
        #temp_mask = np.zeros((256,256))
        #for corner_i in false_corner_id:
        #    temp_mask = cv2.circle(temp_mask, (int(corners[corner_i,1]), int(corners[corner_i,0])), 3, 1.0, -1)
        #plt.imshow(temp_mask)
        #plt.subplot(246)
        #plt.title('false edge: ' + str(len(false_edge_id)))
        #temp_mask = np.zeros((256,256))
        #for edge_i in false_edge_id:
        #    corner1 = np.round(corners[edges[edge_i,0]])
        #    corner2 = np.round(corners[edges[edge_i,1]])
        #    temp_mask = cv2.line(temp_mask, (int(corner1[1]), int(corner1[0])),
        #                         (int(corner2[1]), int(corner2[0])), 1.0, thickness=1)
        #plt.imshow(temp_mask)
        #plt.subplot(247)
        #plt.title('false region: ' + str(len(false_region_id)))
        #temp_mask = np.zeros((256,256))
        #for region_i in false_region_id:
        #    temp_mask += (region_mask == region_i)
        #plt.imshow(temp_mask)
        #plt.subplot(248)
        #plt.title('mask rcnn')
        #temp_mask = np.zeros((256,256))
        #for region_i in range(pred_region_mask.shape[0]):
        #    temp_mask += (region_i+1) * pred_region_mask[region_i]
        #    plt.imshow(temp_mask)
        #plt.show()

        #### debug end ####
        return score_all, score


if __name__ == '__main__':
    from dataset import myDataset
    data_folder = '/local-scratch/fuyang/cities_dataset'

    test_dataset = myDataset(data_folder, phase='valid', edge_linewidth=2, render_pad=-1)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=1,
                                          drop_last=True)

    evaluator = scoreEvaluator({'cornerModelPath':'/local-scratch/fuyang/result/corner_v2/gt_mask_with_gt/models/440.pt',
                            'edgeModelPath':'/local-scratch/fuyang/result/corner_edge_region/edge_graph_drn26_with_search_v2/models/best.pt',
                            'edgeHeatmapPath': '/local-scratch/fuyang/result/corner_edge_region/edge_heatmap_unet/all_edge_masks',
                            'regionHeatmapPath': '/local-scratch/fuyang/result/corner_edge_region/all_region_masks'
                            }, useHeatmap=True)

    for idx, data in enumerate(test_loader):
        name = data['name'][0]
        if name != '1548207062.26':
            continue
        img = data['img'][0]
        graph_data = test_dataset.getDataByName(name)
        conv_data = graph_data['conv_data']
        corners = conv_data['corners']
        edges = conv_data['edges']
        original_score, original_false = evaluator.get_score(img=img, corners=corners, edges=edges, name=name)

