import os 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

batchsize = 16
save_freq = 1
lr_stepsize = 2 
num_epochs = 5
print_freq = 20
graph_per_data = 10
beam_width = 2
beam_depth = 5
epsilon = 0.2

data_folder = './datasets/cities_dataset'
exp_path = './result'
exp_name = 'convmpn_aug' 
save_path = os.path.join(exp_path, exp_name)

lr = 2e-4 

config = {}
for name in list(vars().keys()):
    if name[0] != '_' and name != 'config' and name!='os':
        config[name] = vars()[name]


