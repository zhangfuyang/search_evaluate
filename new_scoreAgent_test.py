import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from new_dataset import myDataset, trainSearchDataset
import random
from new_scoreAgent import scoreEvaluator_with_train
from new_utils import render
import skimage
from drn import drn_c_26

data_folder = '/local-scratch/fuyang/cities_dataset'
beam_width = 6
beam_depth = 10
is_visualize = False
is_save = True
save_path = '/local-scratch/fuyang/result/beam_search_v2/without_search_weak_constraint/'
max_epoch = 100
edge_bin_size = 36
batch_size = 16
phase = 'valid'
edge_strong_constraint = False

train_dataset = trainSearchDataset(data_folder, data_scale=data_scale, phase=phase,
                                   edge_strong_constraint=edge_strong_constraint)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=1,
                                           drop_last=True)

# evaluator_train is used for training
# evaluator_search is used for searching
# separate into two modules in order to use multiple threads to accelerate
evaluator_train = scoreEvaluator_with_train('/local-scratch/fuyang/cities_dataset',
                                            backbone_channel=64, edge_bin_size=edge_bin_size)

evaluator_train.to('cuda:0')
evaluator_train.eval()
evaluator_train.load_weight(save_path, '10')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def test(dataset, model, edge_bin_size):
    correct = 0
    total = 0
    model.eval()
    order = list(range(len(dataset)))
    for count, idx in enumerate(order):
        data = dataset.database[idx]
        name = data['name']
        print(name)
        corners = data['corners']
        edges = data['edges']
        corner_false_id = data['corner_false_id']
        edge_false_id = data['edge_false_id']

        img = skimage.img_as_float(plt.imread(os.path.join(data_folder, 'rgb', name+'.jpg')))
        #img = skimage.transform.rescale(img, self.data_scale, multichannel=True)
        img = img.transpose((2,0,1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]

        mask = render(corners, edges, render_pad=-1)

        ### corner ###
        corner_gt_mask = render(corners[corner_false_id], np.array([]), render_pad=0)[1]

        ###  edge  ###
        edge_gt_mask = render(corners, edges[edge_false_id], render_pad=-1)
        edge_gt_mask = np.concatenate((edge_gt_mask, np.zeros((1,256,256))), 0).transpose((1,2,0))
        edge_input_masks = []
        for edge_i in range(edges.shape[0]):
            edge_input_masks.append(render(corners, edges[[edge_i]],render_pad=-1)[0:1][np.newaxis, ...])
        edge_input_masks = np.concatenate(edge_input_masks, 0)

        img = torch.FloatTensor(img).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        edge_input_masks = torch.FloatTensor(edge_input_masks)

        img = img.to(model.device)
        mask = mask.to(model.device)
        edge_input_masks = edge_input_masks.to(model.device)

        with torch.no_grad():
            img_volume = model.imgvolume(img)
            corner_pred = model.cornerEvaluator(mask, img_volume)
            edge_pred = model.edgeEvaluator(edge_input_masks,
                                            mask.expand(edge_input_masks.shape[0],-1,-1,-1),
                                            img_volume.expand(edge_input_masks.shape[0],-1,-1,-1),
                                            corner_pred.expand(edge_input_masks.shape[0],-1,-1,-1),
                                            torch.zeros(edge_input_masks.shape[0], edge_bin_size, device=model.device))

        # vis
        gt_data = train_dataset.ground_truth[name]
        gt_mask = render(gt_data['corners'], gt_data['edges'])
        gt_mask = np.concatenate((gt_mask, np.zeros((1,256,256))),0).transpose((1,2,0))

        corner_pred = corner_pred.cpu().numpy()
        edge_pred = edge_pred.cpu().numpy()
        edge_pred = np.exp(edge_pred) / np.exp(edge_pred).sum(1, keepdims=True)
        mask = mask.cpu().numpy()

        corner_pred = corner_pred[0,0]
        pred = []
        for edge_i in range(edge_pred.shape[0]):
            if edge_pred[edge_i, 1] > 0.7:
                pred.append(edge_i)

        edge_result = render(corners, edges[pred], render_pad=-1)
        edge_result = np.concatenate((edge_result, np.zeros((1,256,256))), 0).transpose((1,2,0))

        mask = np.concatenate((mask[0], np.zeros((1,256,256))), 0).transpose((1,2,0))
        corner_gt_mask = corner_gt_mask

        # metric
        total += edges.shape[0]
        for edge_i in range(edges.shape[0]):
            if edge_i in edge_false_id and edge_i in pred:
                correct += 1
            elif edge_i not in edge_false_id and edge_i not in pred:
                correct += 1


        # vis
        plt.figure(figsize=(6.4, 3.9))
        img = skimage.img_as_float(plt.imread(os.path.join(data_folder, 'rgb', name+'.jpg')))
        plt.subplot(2,4,1)
        plt.imshow(img)
        plt.subplot(2,4,2)
        plt.imshow(mask)
        plt.subplot(2,4,3)
        plt.imshow(gt_mask)
        plt.subplot(2,4,4)
        plt.imshow(corner_pred)
        plt.title('corner')
        plt.subplot(2,4,5)
        plt.imshow(corner_gt_mask)
        plt.title('corner gt')
        plt.subplot(2,4,6)
        plt.imshow(edge_result)
        plt.title('edge')
        plt.subplot(2,4,7)
        plt.imshow(edge_gt_mask)
        plt.title('edge gt')
        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        #plt.savefig(os.path.join(phase, str(count)+'.jpg'), dpi=300)
        plt.close()

    print(correct / total)


test(train_dataset, evaluator_train, edge_bin_size)

# iter valid  train
#  1   0.63   0.61
#  2   0.70   0.74
#  5   0.72   0.81
# 10   0.74   0.88
# 17   0.74   0.93
# 30   0.75   0.96