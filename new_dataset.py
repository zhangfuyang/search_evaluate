import os
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from new_utils import remove_intersection_and_duplicate, \
    sort_graph, render, get_wrong_corners, get_wrong_edges, simplify_gt, get_corner_bin_map
from new_config import *


class trainSearchDataset(Dataset):
    def __init__(self, datapath, phase='train', data_scale=1.,
                 spatial_match=False, edge_strong_constraint=False,
                 corner_bin=False):
        super(trainSearchDataset, self).__init__()
        self.datapath =datapath
        self.database = []
        self.ground_truth = {}
        self.data_scale = data_scale
        self.spatial_match = spatial_match
        self.edge_strong_constraint = edge_strong_constraint
        self.corner_bin = corner_bin

        name = os.path.join(self.datapath, '{}_list.txt'.format(phase))

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

                conv_data['corners'], conv_data['edges'] = \
                    remove_intersection_and_duplicate(conv_data['corners'], conv_data['edges'], name)

                conv_data['corners'], conv_data['edges'] = sort_graph(conv_data['corners'], conv_data['edges'])
                gt_data = np.load(os.path.join(gt_datapath, name+'.npy'), allow_pickle=True).tolist()
                gt_data['corners'], gt_data['edges'] = sort_graph(gt_data['corners'], gt_data['edges'])

                corners = conv_data['corners']
                edges = conv_data['edges']

                gt_corners = gt_data['corners']
                gt_edges = gt_data['edges']

                self.ground_truth[name] = gt_data
                self.add_data(name, corners, edges)
                self.add_data(name, gt_corners, gt_edges)


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
        #img = skimage.transform.rescale(img, self.data_scale, multichannel=True)
        img = img.transpose((2,0,1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]

        mask = render(corners, edges, render_pad=-1, scale=self.data_scale)

        noise = torch.rand(corners.shape)*4-2  #[-2,2]
        corners = corners + noise.numpy()
        ### corner ###
        corner_gt_mask = render(corners[corner_false_id], np.array([]), render_pad=0, scale=self.data_scale)[1:]

        ###  edge  ###
        id_ = torch.randint(0, edges.shape[0], (1,)).item()
        gt_edge = [1 if id_ in edge_false_id else 0]
        edge_mask = render(corners, edges[[id_]], render_pad=-1, scale=self.data_scale)[0:1]

        ### region ###
        # direct learn segmentation from image, not include in the searching system
        # TODO: could add here as well

        out_data = {}
        if self.corner_bin:
            gt_data = self.ground_truth[name]
            gt_corners = gt_data['corners']
            corner_list_for_each_bin = data['corner_list_for_each_bin']
            bin_map = get_corner_bin_map(gt_corners, corner_list_for_each_bin, bin_size)
            bin_map = torch.FloatTensor(bin_map)
            out_data['bin_map'] = bin_map

        if use_heat_map:
            gt_data = self.ground_truth[name]
            gt_corners = gt_data['corners']
            gt_edges = gt_data['edges']
            heat_map = render(gt_corners, gt_edges, render_pad=0)
            heat_map = torch.FloatTensor(heat_map)
            out_data['gt_heat_map'] = heat_map


        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask)
        corner_gt_mask = torch.FloatTensor(corner_gt_mask)
        gt_edge = torch.LongTensor(gt_edge)
        edge_mask = torch.FloatTensor(edge_mask)

        out_data['img'] = img
        out_data['mask'] = mask
        out_data['corner_gt_mask'] = corner_gt_mask
        out_data['gt_edge'] = gt_edge
        out_data['edge_mask'] = edge_mask
        out_data['name'] = name

        return out_data

    def make_data(self, name, corners, edges):
        gt_data = self.ground_truth[name]
        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            corners, gt_data['corners'], edges, gt_data['edges'])

        gt_corners, gt_edges = simplify_gt(map_same_location, gt_data['corners'], gt_data['edges'])

        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            corners, gt_corners, edges, gt_edges)

        if self.edge_strong_constraint:
            edge_false_id = get_wrong_edges(
                corners, gt_corners, edges, gt_edges,
                map_same_degree)
        else:
            edge_false_id = get_wrong_edges(
                corners, gt_corners, edges, gt_edges,
                map_same_location)

        if self.corner_bin:
            gt_data = self.ground_truth[name]
            gt_corners = gt_data['corners']
            gt_edges = gt_data['edges']

            # get corner dir for each corner
            dirs = []
            for corner_i in range(gt_corners.shape[0]):
                t = np.where(gt_edges==corner_i)
                neighbor_ids = gt_edges[t[0], 1-t[1]]
                neighbor_corner = gt_corners[neighbor_ids]
                the_corner = gt_corners[corner_i]
                vec = neighbor_corner - the_corner
                theta = ((np.arctan2(vec[:,0], vec[:,1])/np.pi+1)*180+bin_size//2) % 360
                the_bin = (theta / bin_size).astype(np.int)
                dirs.append(the_bin)

            corner_list_for_each_bin = []
            for bin_i in range(bin_size):
                cur = []
                for corner_i in range(gt_corners.shape[0]):
                    if bin_i in dirs[corner_i]:
                        cur.append(corner_i)
                corner_list_for_each_bin.append(cur)
        else:
            corner_list_for_each_bin = None

        return {'name': name, 'corners': corners, 'edges': edges,
                              'corner_false_id': list(corner_false_id),
                              'edge_false_id': edge_false_id,
                              'corner_list_for_each_bin':corner_list_for_each_bin}

    def add_processed_data(self, data):
        if len(self.database) >= MAX_DATA_STORAGE:
            del self.database[0]

        self.database.append(data)


    def add_data(self, name, corners, edges):
        return self.add_processed_data(self.make_data(name, corners, edges))




class myDataset(Dataset):
    def __init__(self, datapath, config=None, phase='train',
                 edge_linewidth=2, render_pad=-1, with_gt=False,
                 heat_map=True, raster_match=True, fake_edge=False):
        super(myDataset, self).__init__()
        self.datapath = datapath
        self.phase = phase
        self.config = config
        self.database = []
        self.edge_linewidth = edge_linewidth
        name = os.path.join(self.datapath, phase+'_list.txt')

        with open(name, 'r') as f:
            namelist = f.read().splitlines()

        # load conv-mpn result
        conv_mpn_datapath = os.path.join(self.datapath, 'data/conv-mpn')
        gt_datapath = os.path.join(self.datapath, 'data/gt')
        self.name2id = {}
        print("load conv-mpn result")
        for idx, name in enumerate(namelist):
            if os.path.exists(os.path.join(conv_mpn_datapath, name+'.npy')):
                conv_data = np.load(os.path.join(conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
                conv_data['corners'], conv_data['edges'] = \
                    remove_intersection_and_duplicate(conv_data['corners'], conv_data['edges'], name)
                conv_data['corners'], conv_data['edges'] = sort_graph(conv_data['corners'], conv_data['edges'])
                gt_data = np.load(os.path.join(gt_datapath, name+'.npy'), allow_pickle=True).tolist()
                gt_data['corners'], gt_data['edges'] = sort_graph(gt_data['corners'], gt_data['edges'])
                self.database.append({'conv_data': conv_data, 'gt_data': gt_data,
                                      'name': name, 'corner_data': None, 'edge_data': None,
                                      'region_data': None})
                self.name2id[name] = len(self.database)-1
        print("done.......")

    def __len__(self):
        return len(self.database)

    def getDataByName(self, name):
        return self.database[self.name2id[name]]

    def __getitem__(self, idx):
        name = self.database[idx]['name']
        #conv_data = self.database[idx]['conv_data']
        #gt_data = self.database[idx]['gt_data']
        #corner_data = self.database[idx]['corner_data']
        #edge_data = self.database[idx]['edge_data']
        #region_data = self.database[idx]['region_data']
        #conv_mask = render(conv_data['corners'], conv_data['edges'], self.render_pad, self.edge_linewidth)
        #noise = np.random.random(gt_data['corners'].shape)*3*self.with_gt
        #gt_mask_original = render(gt_data['corners']+noise,
        #                          gt_data['edges'], self.render_pad, self.edge_linewidth)

        img = skimage.img_as_float(plt.imread(os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name+'.jpg')))
        ### test ###
        #plt.subplot(121)
        #plt.imshow(input_edge_mask.transpose(1,2,0))
        #plt.subplot(122)
        #plt.imshow(img)
        #plt.title(str(output_edge_mask))
        #plt.show()

        img = img.transpose((2,0,1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]


        data =  {
            'img': img,
            'name': name}
        return data


if __name__ == '__main__':
    traindataset = trainSearchDataset('/local-scratch/fuyang/cities_dataset', None, data_scale=1)
    for i in range(traindataset.__len__()):
        traindataset.__getitem__(i)


