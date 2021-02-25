import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from utils import remove_intersection_and_duplicate, get_corner_label, render, alpha_blend, get_edge_label, draw_all
import torchvision.transforms as transforms
from PIL import Image
import random 
from threads import searchThread
import time
from new_utils import get_wrong_corners, simplify_gt

import pdb



class CornerDataset(Dataset):
    def __init__(self, datapath, phase='train',
                 edge_linewidth=2, render_pad=-1, with_gt=False,
                 heat_map=True, raster_match=True, fake_edge=False):
        super(CornerDataset, self).__init__()
        self.datapath = datapath
        self.phase = phase
        self.database = []
        self.edge_linewidth = edge_linewidth
        name = os.path.join(self.datapath, phase+'_list.txt')
        with open(name, 'r') as f:
            namelist = f.read().splitlines()
        
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor()])

        
        # Load conv-mpn and gt result
        conv_mpn_datapath = os.path.join(self.datapath, 'data/conv-mpn')
        gt_datapath = os.path.join(self.datapath, 'data/gt')
        per_edge_datapath = os.path.join(self.datapath, 'per_edge')
        self.name2id = {}
        
        print("loading result")
        for idx, name in enumerate(namelist):
            #assert(os.path.exists(os.path.join(conv_mpn_datapath, name+'.npy')))
            #continue
            if os.path.exists(os.path.join(conv_mpn_datapath, name+'.npy')):
                #convmpn_data = np.load(os.path.join(conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
                #convmpn_data['corners'], convmpn_data['edges'] = \
                    #remove_intersection_and_duplicate(convmpn_data['corners'], convmpn_data['edges'], name)
                gt_data = np.load(os.path.join(gt_datapath, name+'.npy'), allow_pickle=True).tolist()
                gt_corners = gt_data['corners']
                gt_edges = gt_data['edges']
                #convmpn_corners = convmpn_data['corners']
                #convmpn_edges = convmpn_data['edges']

                per_edge_data = np.load(os.path.join(per_edge_datapath, name+'.npy'), allow_pickle=True).tolist()
                per_edge_corners = per_edge_data['corners']
                convmpn_corners = np.round(per_edge_corners).astype(np.int)
                convmpn_edges = per_edge_data['edges']


                img = np.array(Image.open(os.path.join(self.datapath, 'rgb', name)+'.jpg'))

                # compute corner labels (location only)
                corner_label, corner_assignment = get_corner_label(gt_corners, convmpn_corners, 7)
                
                # compute edge labels (1: good edge,  0: bad edge)
                edge_label = get_edge_label(convmpn_edges, corner_label, corner_assignment, gt_edges)


                # compute mask and heatmap
                #incorrect_corner_mask = render(convmpn_data['corners'][np.where(corner_label==False)], np.array([]), render_pad=0, scale=1)[1]
                gt_heatmap_large = render(gt_corners, gt_edges, render_pad=0, corner_size=5, edge_linewidth=3)
                gt_heatmap_small = render(gt_corners, gt_edges, render_pad=0, corner_size=3, edge_linewidth=2)
                
                convmpn_mask = render(convmpn_corners, convmpn_edges, 
                            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)

                convmpn_mask_good_edge = render(convmpn_corners, convmpn_edges[edge_label], 
                            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[0]
                convmpn_mask_good_corner = render(convmpn_corners[corner_label], np.array([]), 
                            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[1]
                convmpn_mask_good = np.concatenate([convmpn_mask_good_edge[np.newaxis,:],
                                                    convmpn_mask_good_corner[np.newaxis,:]], axis=0)
                            
                convmpn_mask_bad_edge = render(convmpn_corners, convmpn_edges[~edge_label], 
                            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[0]
                
                edge_masks = []
                for ii in range(convmpn_edges.shape[0]):
                    edge = convmpn_edges[ii][np.newaxis,:]
                    edge_mask = render(convmpn_corners, edge, 
                                       render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[0]
                    edge_masks.append(edge_mask)
                edge_masks = np.array(edge_masks)
                

                convmpn_mask_bad_corner = render(convmpn_corners[~corner_label], np.array([]), 
                            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[1]
                convmpn_mask_bad = np.concatenate([convmpn_mask_bad_edge[np.newaxis,:],
                                                   convmpn_mask_bad_corner[np.newaxis,:]], axis=0)

                self.database.append({'name': name, 'corner_label':corner_label, 'edge_label':edge_label,
                                      'gt_heatmap_small':gt_heatmap_small,'gt_heatmap_large':gt_heatmap_large, 
                                      'convmpn_mask':convmpn_mask, 'convmpn_mask_good':convmpn_mask_good,
                                      'convmpn_mask_bad':convmpn_mask_bad, 'img': img, 'edge_masks':edge_masks})
                self.name2id[name] = len(self.database)-1
            else:
                print('could not find ', name) 
        

    def __len__(self):
        return len(self.database)


    def getDataByName(self, name):
        return self.database[self.name2id[name]]


    def __getitem__(self, idx):
        data = self.database[idx]
        img_norm = self.img_transforms(data['img'])
        gt_heatmap_large = torch.FloatTensor(data['gt_heatmap_large'])   # 0 or 1 
        gt_heatmap_small = torch.FloatTensor(data['gt_heatmap_small'])   # 0 or 1 
        convmpn_mask = torch.FloatTensor(data['convmpn_mask'])  # 0 or 1
        convmpn_mask_good = torch.FloatTensor(data['convmpn_mask_good'])  # 0 or 1
        convmpn_mask_bad = torch.FloatTensor(data['convmpn_mask_bad'])  # 0 or 1
        edge_masks = torch.FloatTensor(data['edge_masks'])
        edge_label = torch.LongTensor(data['edge_label'])
        out_data = {}
        out_data['name'] = data['name']
        out_data['raw_img'] = data['img']
        out_data['img_norm'] = img_norm
        out_data['gt_heatmap_large'] = gt_heatmap_large
        out_data['gt_heatmap_small'] = gt_heatmap_small
        out_data['convmpn_mask'] = convmpn_mask
        out_data['convmpn_mask_good'] = convmpn_mask_good
        out_data['convmpn_mask_bad'] = convmpn_mask_bad
        out_data['edge_masks'] = edge_masks
        out_data['edge_label'] = edge_label
        return out_data



class ReplayMemory(object):
    """Replay memory."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
      
    def push(self, data):
        """Saves a data."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Random sample N data."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EvaluatorDataset(Dataset):
    """Dataloader for training evaluators."""
    def __init__(self, datapath, phase='train', MAX_DATA_STORAGE=5000):
        super(EvaluatorDataset, self).__init__()
        self.datapath = datapath
        self.phase = phase
        self.database = ReplayMemory(MAX_DATA_STORAGE)

        name = os.path.join(self.datapath, phase+'_list.txt')
        with open(name, 'r') as f:
            self.namelist = f.read().splitlines()

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor()])

        self.conv_mpn_datapath = os.path.join(self.datapath, 'data/conv-mpn')
        self.gt_datapath = os.path.join(self.datapath, 'data/gt')
        self.per_edge_datapath = os.path.join(self.datapath, 'per_edge')

        self.name2id = {}
        for idx, name in enumerate(self.namelist):
            self.name2id[name] = idx


    def __len__(self):
        return len(self.database)
        

    def add_all_gt_graph(self):
        """Add all GT graph."""
        print('adding all gt graphs...')
        for idx in range(len(self.namelist)):
            data = self.process_graph(idx, prob_gt=10.0)
            self.database.push(data)


    def add_all_convmpn_graph(self):
        """Add all conv-mpn graph."""
        print('adding all conv-mpn graphs...')
        for idx in range(len(self.namelist)):
            data = self.process_graph(idx, prob_gt=0.0, prob_convmpn=10.0)
            self.database.push(data)


    def process_graph(self, key, new_corners=None, new_edges=None, add_edge_mask=False, prob_gt=0.0, prob_convmpn=0.0):
        """process one graph for adding to memory."""
        if isinstance(key, str):
            name = self.namelist[self.name2id[key]]
        elif isinstance(key, int):
            name = self.namelist[key]
        else:
            raise Exception("Sorry, key must be file name or index")
        
        assert os.path.exists(os.path.join(self.conv_mpn_datapath, name+'.npy'))
                        
        # Load gt corners and edges
        gt_data = np.load(os.path.join(self.gt_datapath, name+'.npy'), allow_pickle=True).tolist()
        gt_corners = gt_data['corners']
        gt_corners = np.round(gt_corners).astype(np.int)
        gt_edges = gt_data['edges']

        # Load convmpn corners and edges
        convmpn_data = np.load(os.path.join(self.conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
        convmpn_data['corners'], convmpn_data['edges'] = \
            remove_intersection_and_duplicate(convmpn_data['corners'], convmpn_data['edges'], name)
        convmpn_corners = convmpn_data['corners']
        convmpn_corners = np.round(convmpn_corners).astype(np.int)
        convmpn_edges = convmpn_data['edges']

        per_edge_data = np.load(os.path.join(self.per_edge_datapath, name+'.npy'), allow_pickle=True).tolist()
        per_edge_corners = per_edge_data['corners']
        per_edge_corners = np.round(per_edge_corners).astype(np.int)
        per_edge_edges = per_edge_data['edges']


        # choose which graph to use
        prob = random.random()
        if prob < prob_gt:    
            add_corners = gt_corners 
            add_edges = gt_edges
        elif prob <  (prob_gt + prob_convmpn):
            add_corners = per_edge_corners#convmpn_corners
            add_edges = per_edge_edges#convmpn_edges
        else:
            assert new_corners is not None
            assert new_edges is not None
            add_corners = new_corners
            add_edges = new_edges

        corner_label, corner_assignment = get_corner_label(gt_corners, add_corners, 7) # corner labels 
        edge_label = get_edge_label(add_edges, corner_label, corner_assignment, gt_edges) # edge labels 

        # compute junction labels
        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            add_corners, gt_corners, add_edges, gt_edges)

        gt_corners, gt_edges = simplify_gt(map_same_location, gt_corners, gt_edges)

        corner_false_id, map_same_degree, map_same_location = get_wrong_corners(
            add_corners, gt_corners, add_edges, gt_edges)

        corner_false_id = list(corner_false_id)
        junction_label = np.ones_like(corner_label)
        junction_label[corner_false_id] = False

        '''
        # All edge masks
        edge_masks = []
        if add_edge_mask:
            for ii in range(add_edges.shape[0]):
                edge = add_edges[ii][np.newaxis,:]
                edge_mask = render(add_corners, edge, 
                    render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[0]
                edge_masks.append(edge_mask)
        edge_masks = np.array(edge_masks)
        '''

        return {'name': name,'add_corners':add_corners, 'add_edges':add_edges, 
                'junction_label':junction_label, 'edge_label': edge_label, 'corner_label':corner_label}


    def __getitem__(self, idx):
        data = self.database.memory[idx]
        name = data['name']

        img = np.array(Image.open(os.path.join(self.datapath, 'rgb', name)+'.jpg')) # load gt image 
        img_norm = self.img_transforms(img)

        gt_data = np.load(os.path.join(self.gt_datapath, name+'.npy'), allow_pickle=True).tolist()
        gt_corners = gt_data['corners']
        gt_corners = np.round(gt_corners).astype(np.int)
        gt_edges = gt_data['edges']

        # GT heatmap
        gt_heatmap = render(gt_corners, gt_edges, 
            render_pad=0, corner_size=3, edge_linewidth=2)

        # Current graph mask
        mask = render(data['add_corners'], data['add_edges'], 
            render_pad=0, scale=1, corner_size=3, edge_linewidth=2) 

        # Target graph mask 
        mask_bad_edge = render(data['add_corners'], data['add_edges'][~data['edge_label']], 
            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[0]
        mask_bad_corner = render(data['add_corners'][~data['corner_label']], np.array([]), 
            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[1]
        target_edgeMask = np.concatenate([mask_bad_edge[np.newaxis,:], mask_bad_corner[np.newaxis,:]], axis=0)
        target_cornerMask = render(data['add_corners'][~data['junction_label']], np.array([]), 
            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[1][np.newaxis,:,:]

        gt_heatmap = torch.FloatTensor(gt_heatmap)   # 0 or 1 
        mask = torch.FloatTensor(mask)  # 0 or 1
        target_edgeMask = torch.FloatTensor(target_edgeMask)  # 0 or 1
        target_cornerMask = torch.FloatTensor(target_cornerMask)  # 0 or 1

        out_data = {}
        out_data['name'] = data['name']
        out_data['raw_img'] = img
        out_data['img_norm'] = img_norm
        out_data['gt_heatmap'] = gt_heatmap
        out_data['mask'] = mask
        out_data['target_edgeMask'] = target_edgeMask
        out_data['target_cornerMask'] = target_cornerMask

        return out_data




def test():
    dataset = EvaluatorDataset(config['data_folder'], phase='train', MAX_DATA_STORAGE=7500)
    dataset.add_all_convmpn_graph()
    #st = searchThread(dataset)
    #st.start()
    dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4)
    for idx, data in enumerate(dataloader):

        raw_img = data['raw_img'][0].detach().cpu().numpy()
        heatmap = data['gt_heatmap'][0].detach().cpu().numpy()
        mask = data['mask'][0].detach().cpu().numpy()
        target_mask = data['target_edgeMask'][0].detach().cpu().numpy()

        f, axarr = plt.subplots(2,2) 
        axarr[0,0].imshow(raw_img)
        axarr[0,0].title.set_text('rgb image')
        axarr[0, 0].axis('off')
        axarr[0,1].imshow(alpha_blend(heatmap[1], heatmap[0]))
        axarr[0,1].title.set_text('gt heatmap')
        axarr[0, 1].axis('off')
        axarr[1,0].imshow(alpha_blend(target_mask[1], target_mask[0]))
        axarr[1,0].title.set_text('target mask')
        axarr[1, 0].axis('off')
        axarr[1,1].imshow(alpha_blend(mask[1], mask[0]))
        axarr[1,1].title.set_text('convmpn mask')
        axarr[1, 1].axis('off')
        #os.makedirs(config['save_path'], exist_ok=True)
        #plt.savefig(os.path.join(config['save_path'], data['name'][0]+'.jpg'), bbox_inches='tight', dpi=150)
        #plt.clf()
        plt.show()
        plt.close()
        
#test()