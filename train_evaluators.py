import os
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from dataset import EvaluatorDataset, myDataset
from threads import searchThread, trainThread
from scoreAgent import scoreEvaluator_with_train as Model
from dicttoxml import dicttoxml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Manager, Lock 


# save configuration file 
print(config)
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')

# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")


# Initialize network
model = Model(config['data_folder'], device1, backbone_channel=64)
with open(os.path.join('./result/pretrained_heatmap', '{}_{}.pt'.format(9, 'heatmapNet')), 'rb') as f:
    model.heatmapNet.load_state_dict(torch.load(f))
model.heatmapNet.eval()
model.to(device1)

searchModel = Model(config['data_folder'], device2, backbone_channel=64)
searchModel.load_state_dict(model.state_dict())
searchModel.to(device2)
searchModel.eval()

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_stepsize'], gamma=0.5)

# Initialize data loader 
train_dataset = EvaluatorDataset(config['data_folder'], phase='train',  edge_strong_constraint=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)
search_dataset = myDataset(config['data_folder'], phase='train')
search_dataloader = torch.utils.data.DataLoader(search_dataset, batch_size=1, shuffle=True, num_workers=1)

manager = Manager()
lock = Lock()
data_memory = manager.list()

tt = trainThread(train_dataset, train_dataloader, model, optimizer, scheduler, device1, searchModel, lock, data_memory)
st = searchThread(search_dataset, search_dataloader, train_dataset, searchModel, data_memory)

tt.start()
st.start()
tt.join()
st.join()






