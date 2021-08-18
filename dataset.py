import os
import skimage
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch
from utils import remove_intersection_and_duplicate, \
    sort_graph, render, get_wrong_corners, get_wrong_edges, simplify_gt, get_corner_bin_map
from config import config
from utils import alpha_blend
import torch.nn.functional as F
from utils import get_corner_label, get_edge_label

MAX_DATA_STORAGE = 20000 #1500



class EvaluatorDataset(Dataset):
    def __init__(self, datapath, phase='train', edge_strong_constraint=False):
        super(EvaluatorDataset, self).__init__()
        self.datapath =datapath
        self.database = []
        self.new_data = []
        self.ground_truth = {}
        self.edge_strong_constraint = edge_strong_constraint

        name = os.path.join(self.datapath, '{}_list.txt'.format(phase))

        with open(name, 'r') as f:
            self.namelist = f.read().splitlines()
        
        # load original result
        self.ip_datapath = os.path.join(self.datapath, 'data/ip')
        self.convmpn_datapath = os.path.join(self.datapath, 'data/conv-mpn')
        self.peredge_datapath = os.path.join(self.datapath, 'data/per_edge')
        self.gt_datapath = os.path.join(self.datapath, 'data/gt')

        for idx, name in enumerate(self.namelist):
            if os.path.exists(os.path.join(self.convmpn_datapath, name+'.npy')):
                conv_data = np.load(os.path.join(self.convmpn_datapath, name+'.npy'), allow_pickle=True).tolist()
                conv_data['corners'], conv_data['edges'] = \
                    remove_intersection_and_duplicate(conv_data['corners'], conv_data['edges'], name)
                #conv_data['corners'] = np.round(conv_data['corners']).astype(int)
                conv_data['corners'], conv_data['edges'] = sort_graph(conv_data['corners'], conv_data['edges'])

                corners = conv_data['corners']
                edges = conv_data['edges']

                gt_data = np.load(os.path.join(self.gt_datapath, name+'.npy'), allow_pickle=True).tolist()
                gt_data['corners'], gt_data['edges'] = sort_graph(gt_data['corners'], gt_data['edges'])

                self.ground_truth[name] = gt_data
                self.add_data(name, corners, edges)

            if os.path.exists(os.path.join(self.ip_datapath, name+'.npy')):
                conv_data = np.load(os.path.join(self.ip_datapath, name+'.npy'), allow_pickle=True).tolist()
                conv_data['corners'] = np.round(conv_data['corners']).astype(int)
                conv_data['corners'], conv_data['edges'] = sort_graph(conv_data['corners'], conv_data['edges'])
 
                corners = conv_data['corners']
                edges = conv_data['edges']

                self.add_data(name, corners, edges)

            if os.path.exists(os.path.join(self.peredge_datapath, name+'.npy')):
                conv_data = np.load(os.path.join(self.peredge_datapath, name+'.npy'), allow_pickle=True).tolist()
                conv_data['corners'] = np.round(conv_data['corners']).astype(int)
                conv_data['corners'], conv_data['edges'] = sort_graph(conv_data['corners'], conv_data['edges'])
 
                corners = conv_data['corners']
                edges = conv_data['edges']

                self.add_data(name, corners, edges)


    def __len__(self):
        #if len(self.new_data) == 0:
            #self.merged_data = self.database
        #else:
            #self.merged_data = self.new_data
        self.merged_data = self.database + self.new_data
        return len(self.merged_data)


    def __getitem__(self, idx):
        data = self.merged_data[idx]
        name = data['name']
        corners = data['corners']
        edges = data['edges']
        corner_false_id = data['corner_false_id']
        edge_false_id = data['edge_false_id']

        img_orig = skimage.img_as_float(plt.imread(os.path.join(self.datapath, 'rgb', name+'.jpg')))
        img = img_orig.transpose((2,0,1))
        img = (img - np.array(config['mean'])[:, np.newaxis, np.newaxis]) / np.array(config['std'])[:, np.newaxis, np.newaxis]

        mask = render(corners, edges, render_pad=-1, scale=1)

        noise = torch.rand(corners.shape)*4-2  #[-2,2]
        corners = corners + noise.numpy()
        
        ### corner ###
        corner_gt_mask = render(corners[corner_false_id], np.array([]), render_pad=0, scale=1)[1:]

        ###  edge  ###
        edge_correct_id = list(set(np.arange(edges.shape[0])) - set(edge_false_id))
        edge_gt_mask = render(corners, edges[list(edge_false_id)], render_pad=0, scale=1)[0:1]

        out_data = {}
       
        gt_data = self.ground_truth[name]
        gt_corners = gt_data['corners']
        gt_edges = gt_data['edges']
        heat_map = render(gt_corners, gt_edges, render_pad=0, corner_size=5, edge_linewidth=3)
        heat_map = torch.FloatTensor(heat_map)
        out_data['gt_heat_map'] = heat_map

        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask)
        corner_gt_mask = torch.FloatTensor(corner_gt_mask)
        edge_gt_mask = torch.FloatTensor(edge_gt_mask)

        out_data['img'] = img
        out_data['mask'] = mask
        out_data['corner_gt_mask'] = corner_gt_mask
        out_data['edge_gt_mask'] = edge_gt_mask
        out_data['name'] = name
        
        '''
        out_data['orig_img'] = img_orig
        # generate edge masks 
        edge_masks = []
        for ii in range(edges.shape[0]):
            edge = edges[ii][np.newaxis,:]
            edge_mask = render(corners, edge, render_pad=0, scale=1)[0:1]
            edge_masks.append(edge_mask)

        #print(avg_edge_len)
        edge_masks = np.array(edge_masks)
        edge_masks = torch.FloatTensor(edge_masks)
        edge_correct_id = torch.LongTensor(edge_correct_id)

        out_data['edge_correct_id'] = edge_correct_id
        out_data['edge_masks'] = edge_masks
        '''

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

        #corner_label, corner_assignment = get_corner_label(gt_corners, corners, 7) # corner labels 
        #edge_label = get_edge_label(edges, corner_label, corner_assignment, gt_edges) # edge labels 
  
        return {'name': name, 'corners': corners, 'edges': edges,
                              'corner_false_id': list(corner_false_id),
                              'edge_false_id': edge_false_id}


    def add_processed_data(self, data):
        #if len(self.database) >= MAX_DATA_STORAGE:
            #del self.database[0]
        self.database.append(data)


    def add_data(self, name, corners, edges):
        return self.add_processed_data(self.make_data(name, corners, edges))


    def _add_processed_data_(self, data):
        if len(self.new_data) >= MAX_DATA_STORAGE:
            del self.new_data[0]
        self.new_data.append(data)


    def _add_data_(self, name, corners, edges):
        return self._add_processed_data_(self.make_data(name, corners, edges))





class myDataset(Dataset):
    def __init__(self, datapath, phase='train'):
        super(myDataset, self).__init__()
        self.datapath = datapath
        self.phase = phase
        self.database = []
        name = os.path.join(self.datapath, phase+'_list.txt')

        with open(name, 'r') as f:
            namelist = f.read().splitlines()

        namelist = namelist[175:]

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
                #conv_data['corners'] = np.round(conv_data['corners']).astype(int)
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
        img = skimage.img_as_float(plt.imread(os.path.join(self.datapath, 'rgb', name+'.jpg')))
        img = img.transpose((2,0,1))
        img = (img - np.array(config['mean'])[:, np.newaxis, np.newaxis]) / np.array(config['std'])[:, np.newaxis, np.newaxis]


        data =  {
            'img': img,
            'name': name}
        return data


def test(): 
    dataset = EvaluatorDataset(config['data_folder'], phase='train',  edge_strong_constraint=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4)
    print(len(dataset))
    for idx, data in enumerate(dataloader):
        #raw_img = data['orig_img'][0].detach().cpu().numpy()
        #heatmap = data['gt_heat_map'][0].detach().cpu().numpy()
        #mask = data['mask'][0].detach().cpu().numpy()
        #corner_mask = data['corner_gt_mask'].squeeze().detach().cpu().numpy()
        #edge_mask = data['edge_gt_mask'].squeeze().detach().cpu().numpy()
        '''
        f, axarr = plt.subplots(2,2) 
        axarr[0,0].imshow(raw_img)
        axarr[0,0].title.set_text('rgb image')
        axarr[0, 0].axis('off')
        axarr[0,1].imshow(alpha_blend(heatmap[1], heatmap[0]))
        axarr[0,1].title.set_text('gt heatmap')
        axarr[0, 1].axis('off')
        axarr[1,0].imshow(alpha_blend(corner_mask, edge_mask))
        axarr[1,0].title.set_text('target mask')
        axarr[1, 0].axis('off')
        axarr[1,1].imshow(alpha_blend(mask[1], mask[0]))
        axarr[1,1].title.set_text('convmpn mask')
        axarr[1, 1].axis('off')
        plt.show()
        plt.close()
        #os.makedirs(config['save_path'], exist_ok=True)
        #plt.savefig(os.path.join(config['save_path'], data['name'][0]+'.jpg'), bbox_inches='tight', dpi=150)
        #plt.clf()
        '''
    
        

#test()
