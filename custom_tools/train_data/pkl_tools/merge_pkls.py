import pickle
import os
import os.path as osp
import mmengine

from tqdm import tqdm

data_train_pth = 'data/train'

data_valid_pth = 'data/valid'

pkls_pth = 'extracted_pkls_related'

# data_train_pth = 'mini_set/train'
# data_valid_pth = 'mini_set/train'


print(f'Train split from {data_train_pth}')
cls_ls = os.listdir(data_train_pth)
train_ls = []
for cls in tqdm(cls_ls):
    cls_vids = [vid for vid in os.listdir(osp.join(data_train_pth, cls)) \
        if vid.endswith('.mp4') or vid.endswith('.avi')]
    
    stripped_cls_vids = [osp.splitext(vid_name)[0] for vid_name in cls_vids]
    train_ls += stripped_cls_vids


print(f'Valid split from {data_valid_pth}')
cls_ls = os.listdir(data_valid_pth)
valid_ls = []
for cls in tqdm(cls_ls):
    cls_vids = [vid for vid in os.listdir(osp.join(data_valid_pth, cls)) \
        if vid.endswith('.mp4') or vid.endswith('.avi')]

    stripped_cls_vids = [osp.splitext(vid_name)[0] for vid_name in cls_vids]
    valid_ls += stripped_cls_vids


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

out = './custom_tools/sub_sample_related.pkl'
mmengine.dump(annotations, out)