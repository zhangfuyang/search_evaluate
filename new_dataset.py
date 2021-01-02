import os
import skimage
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
#from metrics import extract_regions_v2, extract_regions_v2_new
from new_utils import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class myDataset(Dataset):
    def __init__(self, datapath, config=None, phase='train',
                 edge_linewidth=2, render_pad=-1, with_gt=False,
                 heat_map=True, raster_match=True, fake_edge=False):
        super(myDataset, self).__init__()
        self.datapath = datapath
        self.phase = phase
        self.config = config
        self.database = []
        self.with_gt = with_gt
        self.edge_linewidth = edge_linewidth
        self.render_pad = render_pad
        self.heat_map = heat_map
        self.raster_match = raster_match
        self.fake_edge = fake_edge
        name = os.path.join(self.datapath, phase+'_list.txt')

        with open(name, 'r') as f:
            namelist = f.read().splitlines()

        # load conv-mpn result
        conv_mpn_datapath = os.path.join(self.datapath, 'data/conv-mpn')
        gt_datapath = os.path.join(self.datapath, 'data/gt')
        self.name2id = {}
        print("load conv-mpn result")
        for name in namelist:
            if os.path.exists(os.path.join(conv_mpn_datapath, name+'.npy')):
                conv_data = np.load(os.path.join(conv_mpn_datapath, name+'.npy'), allow_pickle=True).tolist()
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






















