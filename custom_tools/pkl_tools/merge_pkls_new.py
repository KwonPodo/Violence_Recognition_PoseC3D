import pickle
import os
import os.path as osp
import mmengine

from tqdm import tqdm

# data_train_pth = 'data/train'
# data_valid_pth = 'data/valid'
data_train_pth = 'custom_tools/train_data/selected_pose_pkls'
data_valid_pth = 'custom_tools/train_data/selected_pose_pkls'

# pkls_pth = 'extracted_pkls_related'
pkls_pth = 'custom_tools/train_data/selected_pose_pkls'


vid_ls = [osp.splitext(pkl)[0] for pkl in sorted(os.listdir(pkls_pth))]
train_ls = vid_ls
valid_ls = vid_ls


annotations = dict()

annotations['split'] = dict()
annotations['split']['train'] = train_ls
annotations['split']['valid'] = valid_ls

# Merge extraced_pkls as 'annotations'
annotations['annotations'] = []

pkls_ls = []
for pkl in tqdm(os.listdir(pkls_pth)):
    if pkl.endswith('.pkl'):
        pkls_ls.append(osp.join(pkls_pth, pkl))

# pkls_ls = [osp.join(pkls_pth, pkl) for pkl in os.listdir(pkls_pth) if pkl.endswith('.pkl')]


for pkl in pkls_ls:
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
        annotations['annotations'].append(data)

out = './custom_tools/train_data/new_dataset_train.pkl'
mmengine.dump(annotations, out)