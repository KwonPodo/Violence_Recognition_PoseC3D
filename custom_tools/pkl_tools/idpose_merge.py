import pickle
import os
import mmengine

parent_dir = 'custom_tools/sample/etri_sample'
vid_ls = [vid for vid in sorted(os.listdir(parent_dir)) if vid.endswith('.mp4')]

pkl_ls = []
for video in vid_ls:
    vid_id = os.path.splitext(os.path.basename(video))[0]
    vid_id_pkl_ls = []
    for pkl in sorted(os.listdir(parent_dir)):
        if pkl.startswith(vid_id) and pkl.endswith('.pkl'):
            vid_id_pkl_ls.append(pkl)
    
    pkl_ls.append(vid_id_pkl_ls)


for pkls in pkl_ls:
    res = []
    unique_id_out = os.path.splitext(pkls[0])[0][:-2] + '.pkl'
    print(pkls)

    for obj_id_pkl in pkls:
        print(obj_id_pkl)
        with open(os.path.join(parent_dir, obj_id_pkl), 'rb') as f:
            data = pickle.load(f)

        frame_kpts = data['keypoint'].squeeze(axis=0)
        frame_kpts_score = data['keypoint_score'].squeeze(axis=0)

        num_frames = frame_kpts.shape[0]
        temporal_kpts_ls = []
        for i in range(num_frames):
            temporal_kpts_ls.append({
                'keypoints' : frame_kpts[i].reshape(1, 17, 2),
                'keypoint_scores' : frame_kpts_score[i].reshape(1, 17)
            })

        res.append(temporal_kpts_ls)
    
    mmengine.dump(res, os.path.join(parent_dir, unique_id_out))
