mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
MAX_DATA_STORAGE = 60000

activate_search_thread = True
search_with_evaluator = True
beam_width = 10  # for search thread
beam_depth = 5  # for search thread
data_scale = 1. #TODO: adapt to scale
edge_bin_size = 36
batch_size = 4
SAFE_NUM = 3
TWO_CORNER_MINIMUM_DISTANCE = 5
score_weights = (1., 2., 100.)

mode = 'strong'
data_folder = '/local-scratch/project/datasets/cities_dataset'
save_path = './result/beam_search_v2/{}_constraint_heatmap_orthogonal/'.format(mode)
#pretrained_path = '/local-scratch/fuyang/result/beam_search_v2/without_search_{}_constraint/'.format(mode)

edge_strong_constraint = True if mode == 'strong' else False
use_heat_map = True
use_bin_map = False
bin_size = 36
use_cross_loss = use_heat_map


config = {}
for name in list(vars().keys()):
    if name[0] != '_' and name != 'config':
        config[name] = vars()[name]
