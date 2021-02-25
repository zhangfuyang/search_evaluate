import os 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

batchsize = 16
save_freq = 5
lr_stepsize = 10
num_epochs = 25
print_freq = 20
beam_width = 3  # for search thread
beam_depth = 2 

search_with_evaluator = False
data_folder = '/local-scratch/xuxiangx/project/datasets/cities_dataset'
exp_path = './result'
exp_name = 'test'
save_path = os.path.join(exp_path, exp_name)
lr = 5e-4

config = {}
for name in list(vars().keys()):
    if name[0] != '_' and name != 'config' and name!='os':
        config[name] = vars()[name]


