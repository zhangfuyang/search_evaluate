import torch
from model import cornerModel, region_model
from drn import drn_c_26
from new_utils import *
import os
import skimage
import matplotlib.pyplot as plt
import threading
import time


class gpu_thread(threading.Thread):
    def __init__(self, threadID, out, sub_update_list):
        threading.Thread.__init__(self)
        self.threadId = threadID
        self.out = out
        self.sub_update_list = sub_update_list

    def run(self):
        start_time = time.time()
        print('[Thread {}] start store edge score'.format(self.threadId))
        self.out = self.out.cpu()
        edge_batch_score = self.out.exp()[:,0] / self.out.exp().sum(1)
        edge_batch_score = edge_batch_score.numpy()
        for edge_i in range(len(self.sub_update_list)):
            score = -1.9*edge_batch_score[edge_i]*edge_batch_score[edge_i]-0.1*edge_batch_score[edge_i]+1
            edge_ele = self.sub_update_list[edge_i][1]
            edge_ele.store_score(score)

        print('[Thread {}] End, in {}s'.format(self.threadId, time.time()-start_time))


class scoreEvaluator():
    def __init__(self, modelPath, useHeatmap=(), useGT=(), dataset=None):
        self.useHeatmap = useHeatmap
        self.useGT = useGT
        self.dataset = dataset
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

        if 'region' in self.useHeatmap:
            self.region_use_heatmap = True
            #self.regionHeatmapPath = modelPath['regionHeatmapPath']
            self.regionHeatmapPath = modelPath['regionEntireMaskPath']
        else:
            self.region_use_heatmap = False

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

    def load_image(self, name):
        img = skimage.img_as_float(plt.imread(os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name+'.jpg')))
        img = img.transpose((2,0,1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = torch.cuda.DoubleTensor(img)
        img = img.unsqueeze(0)
        return img

    def get_fast_score_list(self, candidate_list):
        # load image
        name = candidate_list[0].name
        img = self.load_image(name)

        ########################### corner ################################
        corner_time = time.time()
        batchs = patch_samples(len(candidate_list), 128)
        for batch in batchs:
            inputs = []
            for candidate_i in batch:
                candidate = candidate_list[candidate_i]
                graph = candidate.graph
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
                graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
                graph_mask = torch.cuda.DoubleTensor(graph_mask)
                inputs.append(torch.cat((img, graph_mask), 1))
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat(inputs, 0))
            corner_map = corner_map.detach().cpu().numpy()
            for idx in range(corner_map.shape[0]):
                graph = candidate_list[batch[idx]].graph
                corners = graph.getCornersArray()
                corners_state = self.corner_map2score(corners, corner_map[idx,0])
                corners_score = np.array(corners_state)
                graph.store_score(corner_score=corners_score)

        print('corner time', time.time()-corner_time)
        ############################ edge ##################################
        edge_time = time.time()

        # extract all elements that need recounting
        update_list = []
        for candidate in candidate_list:
            graph = candidate.graph
            edges = graph.getEdges()
            for edge_ele in edges:
                if edge_ele.get_score() is None:
                    update_list.append((graph, edge_ele))

        # split into batches
        batchs = patch_samples(len(update_list), 128)
        gpu2cpu_thread_list = []
        for bi, batch in enumerate(batchs):
            inputs = []
            for update_i in batch:
                update_unit = update_list[update_i]
                graph = update_unit[0]
                edge_ele = update_unit[1]
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
                graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
                graph_mask = torch.cuda.DoubleTensor(graph_mask)
                loc1 = edge_ele.x[0].x
                loc2 = edge_ele.x[1].x
                temp_mask = cv2.line(np.ones((256,256))*-1, loc1[::-1], loc2[::-1],
                                     1.0, thickness=2)[np.newaxis, ...]
                temp_mask = torch.cuda.DoubleTensor(temp_mask).unsqueeze(0)
                inputs.append(torch.cat((img, graph_mask, temp_mask), 1))
            with torch.no_grad():
                out = self.edgeEvaluator(torch.cat(inputs, 0))
            out = out.detach()

            out = out.cpu() # time consuming!!!
            edge_batch_score = out.exp()[:,0] / out.exp().sum(1)
            edge_batch_score = edge_batch_score.numpy()
            for edge_i in range(edge_batch_score.shape[0]):
                score = -1.9*edge_batch_score[edge_i]*edge_batch_score[edge_i]-0.1*edge_batch_score[edge_i]+1
                edge_ele = update_list[batch[edge_i]][1]
                edge_ele.store_score(score)

        print('edge time', time.time()-edge_time)
        ########################## region ###################################
        region_time = time.time()
        gt_mask = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'))

        if 'entire' in self.regionHeatmapPath:
            gt_mask = gt_mask > 0.4
            for candidate in candidate_list:
                graph = candidate.graph
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
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
                regions_score = np.array([iou])

                graph.store_score(region_score=regions_score)

        else:
            for candidate in candidate_list:
                graph = candidate.graph
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
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
                for gt_i in range(gt_mask.shape[0]):
                    best_iou = 0.5
                    best_pred_idx = -1
                    for pred_i in range(len(pred_region_masks)):
                        if pred_i in temp_used_pred:
                            continue
                        iou = IOU(pred_region_masks[pred_i][0], gt_mask[gt_i])
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
                        for gt_i in range(gt_mask.shape[0]):
                            if temp_gt_map[gt_i] != -1:
                                continue
                            iou = IOU(pred_region_masks[pred_i][0], gt_mask[gt_i])
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
                        iou = IOU(pred_region_masks[pred_i][0], gt_mask[gt_idx])
                    regions_state.append(iou*2-1)
                    regions_number.append(pred_region_masks[pred_i][1])

                regions_score = np.array([regions_state])

                graph.store_score(region_score=regions_score)
        print('region', time.time()-region_time)

    def get_fast_score(self, candidate):
        graph = candidate.graph
        corners = graph.getCornersArray()
        edges = graph.getEdgesArray()
        graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
        graph_mask = torch.cuda.DoubleTensor(graph_mask)
        name = candidate.name

        #load image
        img = self.load_image(name)

        #######################   corner   ###############################
        if self.cornerEvaluator is not None:
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat((img, graph_mask), 1))
            corner_map = corner_map.cpu().numpy()[0,0]
            corners_state = self.corner_map2score(corners, corner_map)
            corners_score = np.array(corners_state)
            graph.store_score(corner_score=corners_score)

        #######################   edge     ###############################
        edge_elements = graph.getEdges()
        edge_index = []
        for edge_i in range(len(edge_elements)):
            if edge_elements[edge_i].get_score() is None:
                edge_index.append(edge_i)

        if self.edgeEvaluator is not None:
            batchs = patch_samples(len(edge_index), 8)

            edge_score = np.array([])
            for batch in batchs:
                temp_masks = []
                for edge_i in batch:
                    a = edges[edge_index[edge_i],0]
                    b = edges[edge_index[edge_i],1]
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
            edges_score = np.array(edges_state)
            for idx, edge_i in enumerate(edge_index):
                edge_elements[edge_i].store_score(edges_score[idx])

            #######################   region   ###############################
        if self.region_use_heatmap:
            if 'entire' in self.regionHeatmapPath:
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
                regions_score = np.array([iou])
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

                regions_score = np.array([regions_state])

            graph.store_score(region_score=regions_score)



    def get_score(self, candidate):
        '''
        :param candidate: class Candidate.
        :return:
        '''
        graph = candidate.graph
        corners = graph.getCornersArray()
        edges = graph.getEdgesArray()
        graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
        graph_mask = torch.Tensor(graph_mask).double()
        graph_mask = graph_mask.cuda()
        name = candidate.name

        # load image
        img = self.load_image(name)

        #######################   corner   ###############################
        if self.cornerEvaluator is not None:
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat((img, graph_mask), 1))
            corner_map = corner_map.cpu().numpy()[0,0]
            corners_state = self.corner_map2score(corners, corner_map)
            corners_score = np.array(corners_state)

        #######################   edge     ###############################
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
            edges_score = np.array(edges_state)

        #######################   region   ###############################
        if self.region_use_heatmap:
            if 'entire' in self.regionHeatmapPath:
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
                regions_score = np.array([iou])
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

                regions_score = np.array([regions_state])

        graph.store_score(corner_score=corners_score, edge_score=edges_score,
                          region_score=regions_score)


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
