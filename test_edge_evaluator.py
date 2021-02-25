import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from new_scoreAgent import scoreEvaluator_with_train as Model
from dicttoxml import dicttoxml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time 
from utils import alpha_blend
from new_dataset import EvaluatorDataset

import pdb

# save configuration file 
print(config)
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')

# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0")

# Initialize network
prefix = '25'
model = Model(config['data_folder'], backbone_channel=64)
model.load_weight(config['save_path'], prefix)
model.to(device)
model.eval()

# Initialize data loader 
dataset = EvaluatorDataset(config['data_folder'], phase='valid',  edge_strong_constraint=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Main testing code 
total = 0
total_pos = 0
total_neg = 0
total_correct = 0
total_correct_pos = 0
total_correct_neg = 0

for iteration, data in enumerate(dataloader):
    print(iteration)
    img = data['img'].to(device)
    mask = data['mask'].to(device)
    edge_masks = data['edge_masks'].squeeze().detach().cpu().numpy()#.to(device)
    edge_correct_id = data['edge_correct_id'].squeeze().detach().cpu().numpy()    

    with torch.no_grad():
        img_volume = model.imgvolume(img)  # (bs, 64, 256, 256)
        heatmap_pred = model.getheatmap(img) #(bs, 2, 256, 256)
        corner_pred = model.cornerEvaluator(mask.detach(), img_volume, heatmap_pred.detach()) #(bs, 1, 256, 256)
        edge_pred = model.edgeEvaluator(mask.detach(), img_volume, heatmap_pred.detach()) #(bs, 1, 256, 256)

        gt_mask = mask.squeeze()[0].detach().cpu().numpy()>0

        pred_bad_edges = edge_pred.squeeze().detach().cpu().numpy()
        #pred_bad_edges = np.clip(pred_bad_edges, 0, 1)*gt_mask
        
        for ii in range(edge_masks.shape[0]):
            edge = edge_masks[ii]
            gt_label = ii in edge_correct_id
           
            ratio = np.sum(np.multiply(pred_bad_edges, edge))/np.sum(edge)
            pred_label = ratio < 0.5   # 1: good edge  

            if pred_label == gt_label:
                total_correct += 1
                if gt_label == 1:
                    total_correct_pos += 1
                else:
                    total_correct_neg += 1
            if gt_label == 1:
                total_pos += 1
            else:
                total_neg += 1
            total += 1

        '''
        f, axarr = plt.subplots(1,2) 
        axarr[0].imshow(pred_bad_edges, cmap='gray', vmin=0, vmax=1)
        axarr[0].title.set_text('U-Net Output')
        axarr[0].axis('off')
        axarr[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axarr[1].title.set_text('Conv-MPN')
        axarr[1].axis('off')
        plt.show()
        plt.close()
        '''
        
        
        
       
        
       
print('[TEST] Total Edge Accuracy is {}%, Good Edge Accuracy is {}%, Bad Edge Accuracy is {}%'.format(
    round(100.0*total_correct/total, 5),
    round(100.0*total_correct_pos/total_pos, 5),
    round(100.0*total_correct_neg/total_neg, 5)))
'''
print(total_correct_pos)
print(total_pos)
print(total_correct_neg)
print(total_neg)
print(total_correct)
print(total)
'''
