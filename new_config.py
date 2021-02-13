mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
MAX_DATA_STORAGE = 60000
MEMORY_SIZE = 17000
step_size = 4000
max_run = 6
SAMPLE_UPDATE = 5
sample_edges = 3
activate_search_thread = True
beam_width = 1  # for search thread
beam_depth = 5  # for search thread
data_scale = 1. #TODO: adapt to scale
edge_bin_size = 36
epsilon = 0.3
batch_size = 4
TARGET_UPDATE = 50
MAX_Q_SIZE = 1
gamma = 0.1
SAFE_NUM = 3
TWO_CORNER_MINIMUM_DISTANCE = 5
score_weights = (1., 2., 100.)
SAVE_FREQ = 400

base_path = '/local-scratch/project/datasets'  #/local-scratch/fuyang'  
mode = 'strong' # use edge strong constraint (need direction and degree both correct), otherwise, only based on the location. Default is 'strong'
edge_strong_constraint = True if mode == 'strong' else False
data_folder = base_path + '/cities_dataset'
save_path = './result/gamma0.1/{}_constraint_heatmap_orthogonal/'.format(mode) #'/local-scratch/fuyang/result/beam_search_v2/{}_constraint_heatmap_orthogonal/'.format(mode)
#pretrained_path = '/local-scratch/fuyang/result/beam_search_v2/without_search_{}_constraint/'.format(mode)

use_heat_map = True  # predict heatmap as well, Default True
use_bin_map = False # deprecated
bin_size = 36 # deprecated
use_cross_loss = use_heat_map & True # use heatmap pseudo loss, if use, use_heat_map must be set as True


config = {}
for name in list(vars().keys()):
    if name[0] != '_' and name != 'config':
        config[name] = vars()[name]
