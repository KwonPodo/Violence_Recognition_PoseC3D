# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
import argparse
import os
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory

import mmengine
import numpy as np
from tqdm import tqdm
from pprint import pprint

from mmaction.apis import detection_inference, pose_inference
from mmaction.utils import frame_extract

np.set_printoptions(suppress=True)

# args = abc.abstractproperty()
# args.det_config = 'demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501
# args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
# args.det_score_thr = 0.5
# args.pose_config = 'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501
# args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501

from abc import ABC, abstractmethod

class Args(ABC):
    @property
    @abstractmethod
    def det_config(self):
        pass

    @property
    @abstractmethod
    def det_checkpoint(self):
        pass

    @property
    @abstractmethod
    def det_score_thr(self):
        pass

    @property
    @abstractmethod
    def pose_config(self):
        pass

    @property
    @abstractmethod
    def pose_checkpoint(self):
        pass

class ConcreteArgs(Args):
    @property
    def det_config(self):
        return 'demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'

    @property
    def det_checkpoint(self):
        return 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

    @property
    def det_score_thr(self):
        return 0.5

    @property
    def pose_config(self):
        return 'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'

    @property
    def pose_checkpoint(self):
        return 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

args = ConcreteArgs()


def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)

def pose_inference_with_align(args, frame_paths, det_results):
    # filter frame without det bbox
    det_results = [
        frm_dets for frm_dets in det_results if frm_dets.shape[0] > 0
    ]

    pose_results, _ = pose_inference(args.pose_config, args.pose_checkpoint,
                                     frame_paths, det_results, args.device)
    # align the num_person among frames
    num_persons = max([pose['keypoints'].shape[0] for pose in pose_results])
    num_points = pose_results[0]['keypoints'].shape[1]
    num_frames = len(pose_results)
    keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                         dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
            scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

    return keypoints, scores

def read_track_ann(fname=None):
    assert fname is not None

    # {frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}, {needed}
    # frame_id, obj_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], conf_score, related

    print(f'====> Reading {fname}')

    with open(fname, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        lines[i] = list(map(float, line.split(',')))
        lines[i][0] = int(lines[i][0]) # frame_id
        lines[i][1] = int(lines[i][1]) # obj_id
        lines[i][-1] = int(lines[i][-1]) # related

    return lines

def track_postproc(vid, track_results):
    # Track Results : [[frame_id, obj_id, x, y, w, h, score, related]]
    # : List[np.ndarray(obj_num, 4)] -> List length == frame_num
    # [M, T, K, V] => [1, Frame, 17, 2]
    tmp_track_res = np.array(track_results)

    total_frame = int(tmp_track_res[:,0].max(axis=0))
    num_objs = int(tmp_track_res[:,1].max(axis=0))

    print(f'total_frame : {total_frame}')

    # Get only related ids
    obj_id_set = set()
    new_track_results = []
    for track_arr in track_results:
        frame_id, obj_id, tlx, tly, w, h, score, related = track_arr

        if related:
            obj_id_set.add(obj_id)
            new_track_results.append(track_arr[:-2])
    

    related_num_objs = len(list(obj_id_set))
    print(f'Related_num_objs : {related_num_objs}')

    track_id_map = dict()
    for i, id in enumerate(list(obj_id_set)):
        track_id_map[id] = i
    
    print(f'Track_id_remap : {track_id_map}')

    related_track_ls = [np.zeros((total_frame, 4)) for i in range(related_num_objs)]

    for track_arr in new_track_results:
        frame_id, obj_id, tlx, tly, w, h = track_arr

        rate = 1.25
        enlarged_w = w * rate
        enlarged_h = h * rate
        enlarged_w_half = enlarged_w / 2.
        enlarged_h_half = enlarged_h / 2.

        tlc_x = tlx + w/2.
        tlc_y = tly + h/2.

        new_tlx = tlc_x - enlarged_w_half
        new_tly = tlc_y - enlarged_h_half

        entry = np.array([new_tlx, new_tly, new_tlx + enlarged_w, new_tly + enlarged_h])

        idx = track_id_map[obj_id]
        related_track_ls[idx][frame_id-1] = entry
    
    print(f'related_track_ls len : {len(related_track_ls)}')

    # final_ls = np.split(related_track_ls[0], total_frame, axis=0)
    final_ls = []

    # (598, 4) 배열을 (1, 4) 배열로 분할하여 리스트로 변환
    for obj_tracked in related_track_ls:
        sub_ls = [obj_tracked[i:i+1,:] for i in range(obj_tracked.shape[0])]
        final_ls.append(sub_ls)
    
    return final_ls
    
def bytetrack_pose_extraction(vid):
    tmp_dir = TemporaryDirectory()
    frame_paths, _ = frame_extract(vid, out_dir=tmp_dir.name)

    '''
    det_results = List[np.ndarray] of len == total_frames
                       ㄴ np.ndarray has shape of [? , 4] 'xyxy' ? 'xywh'
                            ㄴ ? == num of objs
    '''
    
    root_pth = osp.dirname(vid)
    ann_name = osp.basename(vid) + '_related.txt'
    ann_pth = osp.join(root_pth, ann_name)

    track_results = read_track_ann(ann_pth) # ByteTrack Tracking 결과들을 읽어옴
    track_results = track_postproc(vid, track_results) # Related track_id만 뽑아 id remapping
    print(type(track_results), len(track_results))
    # pprint(track_results)
    # print(track_results[0].shape)

    anno_ls = []

    for i, result in enumerate(track_results):
        anno = dict()
        keypoints, scores = pose_inference_with_align(args, frame_paths,
                                                    result)

        anno['keypoint'] = keypoints
        anno['keypoint_score'] = scores
        anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
        anno['img_shape'] = (1080, 1920)
        anno['original_shape'] = (1080, 1920)
        anno['total_frames'] = keypoints.shape[1]
        anno['label'] = int(osp.basename(vid).split('_')[-1][:2])

        anno_ls.append(anno)

    tmp_dir.cleanup()

    return anno_ls

def extract_from_path(root_pth):
    cls_dirs = os.listdir(root_pth)
    for dir in tqdm(cls_dirs):
        dir_pth = osp.join(root_pth, dir)
        print(f'==> Working on directory {dir_pth}')

        vid_ls = [vid for vid in os.listdir(dir_pth) if vid.endswith('.mp4') or vid.endswith('.avi')]
        ann_ls = [ann for ann in os.listdir(dir_pth) if ann.endswith('.txt')]

        for vid in vid_ls:
            vid_pth = osp.join(dir_pth, vid)
            print(f'====> Working on Video {vid_pth}')

            anno_ls = bytetrack_pose_extraction(vid_pth)

            for i, anno in enumerate(anno_ls):
                pkl_name = f'{osp.basename(vid)}_{i}.pkl'
                anno_pkl_dump_pth = osp.join(args.out_dir, pkl_name)

                print(f'=======> dumping at {anno_pkl_dump_pth}')
                mmengine.dump(anno, anno_pkl_dump_pth)

        print(f'Finished Directory : {dir}')
        print('#'*50)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single video')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--single_dir', action='store_true')
    parser.add_argument('--single_file', action='store_true')

    parser.add_argument('--out-dir', type=str, default='extracted_pkls_related')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.out_dir = global_args.out_dir

    if global_args.single_dir:
        # Single Directory Pose Extraction
        dir_pth = './sub_sample/train/raise'
        print(f'==> Working on directory {dir_pth}')

        vid_ls = [vid for vid in os.listdir(dir_pth) if vid.endswith('.mp4') or vid.endswith('.avi')]
        ann_ls = [ann for ann in os.listdir(dir_pth) if ann.endswith('.txt')]

        for vid in vid_ls:
            vid_pth = osp.join(dir_pth, vid)
            print(f'====> Working on Video {vid_pth}')

            anno = bytetrack_pose_extraction(vid_pth)
            pkl_name = osp.basename(vid) + '.pkl'
            anno_pkl_dump_pth = osp.join(args.out_dir, pkl_name)

            print(anno_pkl_dump_pth)
            mmengine.dump(anno, anno_pkl_dump_pth)


        # Single File Pose Extraction
    elif global_args.single_file:
        # Single File Pose Extraction
        file_pth = 'data/train/fight/1_069_1_04.mp4'
        print(f'==> Working on file {file_pth}')

        anno_ls = bytetrack_pose_extraction(file_pth)

        for i, anno in enumerate(anno_ls):
            pkl_name = f'{osp.basename(file_pth)}_{i}.pkl'
            anno_pkl_dump_pth = osp.join(args.out_dir, pkl_name)
            mmengine.dump(anno, anno_pkl_dump_pth)
            print(f'Saved pickle file to {anno_pkl_dump_pth}')
    else: # Sub_sample Pose Extraction
        print("Train/Val")

        data_train_pth = './data/train'
        data_val_pth = './data/valid'

        # Extract from Train split
        cls_dirs = os.listdir(data_train_pth)
        for dir in cls_dirs:
            dir_pth = osp.join(data_train_pth, dir)
            print(f'==> Working on directory {dir_pth}')

            vid_ls = [vid for vid in os.listdir(dir_pth) if vid.endswith('.mp4') or vid.endswith('.avi')]
            ann_ls = [ann for ann in os.listdir(dir_pth) if ann.endswith('.txt')]

            for vid in vid_ls:
                vid_pth = osp.join(dir_pth, vid)
                print(f'====> Working on Video {vid_pth}')

                anno_ls = bytetrack_pose_extraction(vid_pth)

                for i, anno in enumerate(anno_ls):
                    pkl_name = f'{osp.basename(vid)}_{i}.pkl'
                    anno_pkl_dump_pth = osp.join(args.out_dir, pkl_name)

                    print(f'=======> dumping at {anno_pkl_dump_pth}')
                    mmengine.dump(anno, anno_pkl_dump_pth)
            
            print(f'Finished Directory : {dir}')
            print('#'*50)

        # Extract from Valid split
        cls_dirs = os.listdir(data_val_pth)
        for dir in tqdm(cls_dirs):
            dir_pth = osp.join(data_val_pth, dir)
            print(f'==> Working on directory {dir_pth}')

            vid_ls = [vid for vid in os.listdir(dir_pth) if vid.endswith('.mp4') or vid.endswith('.avi')]
            ann_ls = [ann for ann in os.listdir(dir_pth) if ann.endswith('.txt')]

            for vid in vid_ls:
                vid_pth = osp.join(dir_pth, vid)
                print(f'====> Working on Video {vid_pth}')

                anno_ls = bytetrack_pose_extraction(vid_pth)

                for i, anno in enumerate(anno_ls):
                    pkl_name = f'{osp.basename(vid)}_{i}.pkl'
                    anno_pkl_dump_pth = osp.join(args.out_dir, pkl_name)

                    print(f'=======> dumping at {anno_pkl_dump_pth}')
                    mmengine.dump(anno, anno_pkl_dump_pth)

            print(f'Finished Directory : {dir}')
            print('#'*50)

