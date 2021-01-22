import os
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
from drn import drn_c_26
from new_utils import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class trainSearchDataset(Dataset):
    def __init__(self, datapath, data_scale=1., spatial_match=False):
        super(trainSearchDataset, self).__init__()
        self.datapath =datapath
        self.database = []
        self.ground_truth = {}
        self.data_scale = data_scale
        self.spatial_match = spatial_match

        name = os.path.join(self.datapath, 'train_list.txt')

        with open(name, 'r') as f:
            namelist = f.read().splitlines()

        # load original result
        conv_mpn_datapath = os.path.join(self.datapath, 'data/conv-mpn')
        gt_datapath = os.path.join(self.datapath, 'data/gt')

        for idx, name in enumerate(namelist):
            #if idx == 32:
            #    break
            if os.path.exists(os.path.join(conv_mpn_datapath, name+'.npy')):
                conv_data = np.load(os.path.join(conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
                conv_data['corners'], conv_data['edges'] = sort_graph(conv_data['corners'], conv_data['edges'])
                gt_data = np.load(os.path.join(gt_datapath, name+'.npy'), allow_pickle=True).tolist()
                gt_data['corners'], gt_data['edges'] = sort_graph(gt_data['corners'], gt_data['edges'])

                #TODO: if intersection add the x junction and split edges
                corners = conv_data['corners']
                edges = conv_data['edges']

                self.ground_truth[name] = gt_data
                self.add_data(name, corners, edges)

                #conv_mask1 = render(conv_data['corners'], conv_data['edges'])
                #conv_mask2 = render(conv_data['corners'][list(corner_false_id)], np.array([]))[1:,:,:]
                #conv_mask = np.concatenate((conv_mask1,conv_mask2), 0).transpose(1,2,0)

                #gt_mask1 = render(gt_data['corners'], gt_data['edges'])
                #gt_mask2 = np.zeros((1,256,256))
                #gt_mask = np.concatenate((gt_mask1, gt_mask2), 0).transpose(1,2,0)

                #conv_edge = render(conv_data['corners'], conv_data['edges'][edge_false_id])[0:1,:,:]
                #conv_edge = np.concatenate((conv_mask1,conv_edge), 0).transpose(1,2,0)

                #plt.subplot(131)
                #plt.imshow(conv_mask)
                #plt.subplot(132)
                #plt.imshow(conv_edge)
                #plt.subplot(133)
                #plt.imshow(gt_mask)
                #plt.show()

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        data = self.database[idx]
        name = data['name']
        corners = data['corners']
        edges = data['edges']
        corner_false_id = data['corner_false_id']
        edge_false_id = data['edge_false_id']

        img = skimage.img_as_float(plt.imread(os.path.join(self.datapath, 'rgb', name+'.jpg')))

        mask = render(corners, edges, render_pad=-1, scale=self.data_scale)
        mask = np.concatenate((mask, np.zeros((1,256,256))), 0).transpose((1,2,0))

        gt_data = self.ground_truth[name]
        gt_mask = render(gt_data['corners'], gt_data['edges'], render_pad=-1, scale=self.data_scale)
        gt_mask = np.concatenate((gt_mask, np.zeros((1,256,256))), 0).transpose((1,2,0))

        ### corner ###
        corner_mask = render(corners[corner_false_id], np.array([]), render_pad=0, scale=self.data_scale)[1]

        ###  edge  ###
        edge_mask = render(corners, edges[edge_false_id], render_pad=-1, scale=self.data_scale)
        edge_mask = np.concatenate((edge_mask, np.zeros((1,256,256))), 0).transpose((1,2,0))



        plt.subplot(2,3,1)
        plt.imshow(img)
        plt.subplot(2,3,2)
        plt.imshow(gt_mask)
        plt.subplot(2,3,3)
        plt.imshow(mask)
        plt.subplot(2,3,4)
        plt.imshow(corner_mask)
        plt.subplot(2,3,5)
        plt.imshow(edge_mask)
        plt.show()


    def add_data(self, name, corners, edges):
        gt_data = self.ground_truth[name]
        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            corners, gt_data['corners'], edges, gt_data['edges'])

        edge_false_id = get_wrong_edges(
            corners, gt_data['corners'], edges, gt_data['edges'],
            map_same_location)

        self.database.append({'name': name, 'corners': corners, 'edges': edges,
                              'corner_false_id': list(corner_false_id),
                              'edge_false_id': edge_false_id})



if __name__ == '__main__':
    traindataset = trainSearchDataset('/local-scratch/fuyang/cities_dataset', data_scale=1.)
    for i in range(traindataset.__len__()):
        traindataset.__getitem__(i)

