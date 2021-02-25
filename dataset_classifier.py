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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import random 


import pdb


os.makedirs(config['save_path'], exist_ok=True)

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
        self.name2id = {}
        print("loading result")
        for idx, name in enumerate(namelist):
            if os.path.exists(os.path.join(conv_mpn_datapath, name+'.npy')):
                convmpn_data = np.load(os.path.join(conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
                convmpn_data['corners'], convmpn_data['edges'] = \
                    remove_intersection_and_duplicate(convmpn_data['corners'], convmpn_data['edges'], name)
                gt_data = np.load(os.path.join(gt_datapath, name+'.npy'), allow_pickle=True).tolist()

                gt_corners = gt_data['corners']
                gt_edges = gt_data['edges']
                convmpn_corners = convmpn_data['corners']
                convmpn_edges = convmpn_data['edges']

                img = np.array(Image.open(os.path.join(self.datapath, 'rgb', name)+'.jpg'))

                # compute corner labels for conv-mpn 
                corner_label, corner_assignment = get_corner_label(gt_corners, convmpn_corners, 7)
                
                # compute edge labels for conv-mpn (1: good edge,  0: bad edge)
                edge_label = get_edge_label(convmpn_edges, corner_label, corner_assignment, gt_edges)

                # compute mask and heatmap
                #incorrect_corner_mask = render(convmpn_data['corners'][np.where(corner_label==False)], np.array([]), render_pad=0, scale=1)[1]
                gt_heatmap_large = render(gt_corners, gt_edges, render_pad=0, corner_size=5, edge_linewidth=3)
                gt_heatmap_small = render(gt_corners, gt_edges, render_pad=0, corner_size=3, edge_linewidth=2)
                
                convmpn_mask = render(convmpn_corners, convmpn_edges, 
                            render_pad=0, scale=1, corner_size=3, edge_linewidth=2)

                self.database.append({'name': name, 'corner_label':corner_label, 'edge_label':edge_label,
                                      'gt_heatmap_small':gt_heatmap_small,'gt_heatmap_large':gt_heatmap_large, 
                                      'convmpn_mask':convmpn_mask, 'convmpn_edges':convmpn_edges,
                                      'img': img, 'convmpn_corners':convmpn_corners})
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
        convmpn_edges = data['convmpn_edges']
        convmpn_corners = data['convmpn_corners']
        edge_label = torch.LongTensor(data['edge_label'])

        good_edge = np.where(edge_label==True)[0].tolist()
        bad_edge = np.where(edge_label==False)[0].tolist()
        sample_size = 2
        if len(good_edge) < sample_size: 
            ids = random.sample(bad_edge, sample_size) #random.choice(edge_false_id)
        elif len(bad_edge) < sample_size: 
            ids = random.sample(good_edge, sample_size) #random.choice(edge_correct_id)
        else:
            correct_id = random.sample(good_edge, sample_size) #random.choice(edge_correct_id)
            false_id = random.sample(bad_edge, sample_size) #random.choice(edge_false_id)
            ids = []
            for ii in range(sample_size):    
                if random.random() > 0.5:
                    id_ = correct_id[ii]
                else:
                    id_ = false_id[ii]
                ids.append(id_)
        #ids = random.sample(list(np.arange(convmpn_edges.shape[0])), 1)
        edge_masks = []
        for ii in range(convmpn_edges.shape[0]):#ids:
            edge = convmpn_edges[ii][np.newaxis,:]
            edge_mask = render(convmpn_corners, edge, render_pad=0, scale=1, corner_size=3, edge_linewidth=2)[0]
            edge_masks.append(edge_mask)
        edge_masks = np.array(edge_masks)

        edge_masks = torch.FloatTensor(edge_masks)
        edge_label = edge_label#[ids]#[ids]

        out_data = {}
        out_data['name'] = data['name']
        out_data['raw_img'] = data['img']
        out_data['img_norm'] = img_norm
        out_data['gt_heatmap_large'] = gt_heatmap_large
        out_data['gt_heatmap_small'] = gt_heatmap_small
        out_data['convmpn_mask'] = convmpn_mask
        out_data['edge_masks'] = edge_masks
        out_data['edge_label'] = edge_label

        return out_data




def test():
    dataset = CornerDataset(config['data_folder'], phase='valid', edge_linewidth=2, render_pad=-1)
    dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4)
    for idx, data in enumerate(dataloader):
        print(idx)
        pdb.set_trace()
        continue
        raw_img = data['raw_img'][idx].detach().cpu().numpy()
        heatmap_large = data['gt_heatmap_large'][idx].detach().cpu().numpy()
        heatmap_small = data['gt_heatmap_small'][idx].detach().cpu().numpy()
        convmpn_mask = data['convmpn_mask'][idx].detach().cpu().numpy()
        incorrect_corner_mask = data['incorrect_corner_mask'][idx].detach().cpu().numpy()

        f, axarr = plt.subplots(2,2) 
        axarr[0,0].imshow(raw_img)
        axarr[0,0].title.set_text('rgb image')
        axarr[0, 0].axis('off')
        axarr[0,1].imshow(alpha_blend(heatmap[1], heatmap[0]))
        axarr[0,1].title.set_text('gt heatmap')
        axarr[0, 1].axis('off')
        axarr[1,0].imshow(alpha_blend(incorrect_corner_mask, convmpn_mask[0]))
        axarr[1,0].title.set_text('bad corner')
        axarr[1, 0].axis('off')
        axarr[1,1].imshow(alpha_blend(convmpn_mask[1], convmpn_mask[0]))
        axarr[1,1].title.set_text('convmpn mask')
        axarr[1, 1].axis('off')
        os.makedirs(config['save_path'], exist_ok=True)
        plt.savefig(os.path.join(config['save_path'], data['name'][0]+'.jpg'), bbox_inches='tight', dpi=150)
        plt.clf()

#test()