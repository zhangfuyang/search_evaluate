import os
import numpy as np
import pickle
from new_utils import *

base_path = '/local-scratch/fuyang/result/beam_search_v2/strong_constraint_heatmap_without_corner_heatmap/valid_prefix_4_result'
datapath = '/local-scratch/fuyang/cities_dataset'
#obj_name = 'iter_0_num_0.obj'
obj_name = 'best_0.obj'
post_process = False

gt_datapath = os.path.join(datapath, 'data/gt')

metric = Metric()
corner_tp = 0.0
corner_fp = 0.0
corner_length = 0.0
edge_tp = 0.0
edge_fp = 0.0
edge_length = 0.0
region_tp = 0.0
region_fp = 0.0
region_length = 0.0

def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp+fp+1e-8)
    return recall, precision

for file_name in os.listdir(base_path):
    if len(file_name) < 10:
        continue
    f = open(os.path.join(base_path, file_name, obj_name), 'rb')
    gt_data = np.load(os.path.join(gt_datapath, file_name+'.npy'), allow_pickle=True).tolist()
    candidate = pickle.load(f)
    conv_corners = candidate.graph.getCornersArray()
    conv_edges = candidate.graph.getEdgesArray()
    conv_data = {'corners': conv_corners, 'edges': conv_edges}
    score = metric.calc(gt_data, conv_data)
    corner_tp += score['corner_tp']
    corner_fp += score['corner_fp']
    corner_length += score['corner_length']
    edge_tp += score['edge_tp']
    edge_fp += score['edge_fp']
    edge_length += score['edge_length']
    region_tp += score['region_tp']
    region_fp += score['region_fp']
    region_length += score['region_length']

f = open(os.path.join(base_path, 'score.txt'), 'w')
# corner
recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
f_score = 2.0*precision*recall/(recall+precision+1e-8)
print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
f.write('corners - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

# edge
recall, precision = get_recall_and_precision(edge_tp, edge_fp, edge_length)
f_score = 2.0*precision*recall/(recall+precision+1e-8)
print('edges - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
f.write('edges - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

# region
recall, precision = get_recall_and_precision(region_tp, region_fp, region_length)
f_score = 2.0*precision*recall/(recall+precision+1e-8)
print('regions - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
f.write('regions - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

f.close()





