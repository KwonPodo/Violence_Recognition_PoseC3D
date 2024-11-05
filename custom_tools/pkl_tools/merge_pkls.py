import pickle
import os
import os.path as osp
import mmengine

from tqdm import tqdm

data_train_pth = 'data/Nov_1_dataset/all/extracted_pose_pkls'
data_valid_pth = 'data/Nov_1_dataset/all/test_set/extracted_pose_pkls'
merged_out_path = 'data/Nov_1_dataset/all/train+val.pkl'

train_ls = [osp.splitext(pkl)[0] for pkl in sorted(os.listdir(data_train_pth))]
valid_ls = [osp.splitext(pkl)[0] for pkl in sorted(os.listdir(data_valid_pth))]

train_ls.extend(valid_ls)

print(f'train : {len(train_ls)}')
print(f'valid : {len(valid_ls)}')

annotations = dict()
annotations['split'] = dict()
annotations['split']['train'] = train_ls
annotations['split']['valid'] = valid_ls
# annotations['split']['test'] = valid_ls

# Merge extracted_pkls as 'annotations'
annotations['annotations'] = []

# train 
pkls_ls = [os.path.join(data_train_pth, pkl)
           for pkl in sorted(os.listdir(data_train_pth)) if pkl.endswith('.pkl')]

# valid
pkls_ls.extend([os.path.join(data_valid_pth, pkl)
                for pkl in sorted(os.listdir(data_valid_pth)) if pkl.endswith('.pkl')])



for pkl in pkls_ls:
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
        annotations['annotations'].append(data)


mmengine.dump(annotations, merged_out_path)
