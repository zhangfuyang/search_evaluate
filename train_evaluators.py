import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from new_dataset import EvaluatorDataset, myDataset
from threads import searchThread, trainThread, searchThread_preprocess
from new_scoreAgent import scoreEvaluator_with_train as Model
from dicttoxml import dicttoxml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time 
from multiprocessing import Process, Lock, Manager, Pool

import pdb



# save configuration file 
print(config)
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')

# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

# Initialize network
model = Model(config['data_folder'], backbone_channel=64)
model.to(device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_stepsize'], gamma=0.5)

# Initialize data loader 
train_dataset = EvaluatorDataset(config['data_folder'], phase='train',  edge_strong_constraint=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)
#dataset.add_all_convmpn_graph()
#dataset.add_all_gt_graph()

#search_dataset = myDataset(config['data_folder'], phase='train')
#search_dataloader = torch.utils.data.DataLoader(search_dataset, batch_size=1, shuffle=True, num_workers=1)




manager = Manager()
shared_buffer = manager.list() # shared buffer
lock = Lock()

print('Non-augmented dataset size: {}'.format(len(train_dataset)))
print('Start searching in parallel...')
pool = Pool(processes=50)
for idx, name in enumerate(train_dataset.namelist):
    pool.apply_async(searchThread_preprocess, args=(name, shared_buffer, ))
pool.close()
pool.join()

print('Augmented dataset size: {}'.format(len(shared_buffer)))
print('Start training with augmented data...')
trainThread(train_dataset, train_dataloader, model, optimizer, scheduler, device, shared_buffer, lock) # start train process











