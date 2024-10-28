import argparse
import copy as cp
import tempfile
import pickle
import os
import os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple, Union
from pprint import pprint
from tqdm import tqdm

from torch import nn
import numpy as np
import mmengine
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress

from mmaction.apis import init_recognizer
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Extraction for VD training')
    group = parser.add_mutually_exclusive_group(required=True)

    # Exclusive Group Arguments
    group.add_argument(
        '-f', '--file',
        action='store_true',
        help='Extract pose from a single file.'
    )
    group.add_argument(
        '-d', '--directory',
        action='store_true',
        help='Extract pose from files in a single directory.'
    )
    group.add_argument(
        '-p', '--parent-dirs',
        action='store_true',
        help='Extract pose from dirs below parent directory.'
    )
    group.add_argument(
        '-m', '--multiple-dirs',
        action='store_true',
        help='Extract pose from multiple directories.'
    )

    # Common Arguments
    parser.add_argument(
        'paths',
        nargs='+',
        help='Video & Bbox Paths(s) to extract pose based on the selected mode.'
    )
    parser.add_argument(
        '--out-root',
        default='custom_tools/train_data/pose_pkls',
        help=''
    )
    parser.add_argument(
        '--pose-config',
        default='pipeline_integration/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='HRNet Config file path (from mmpose)'
    )
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='HRNet-w32 Checkpoint file/url'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option'
    )

    args = parser.parse_args()

    return args, parser


def init_pose_model(args):
    try:
        from mmpose.apis import init_model
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`.')
    
    if isinstance(args.pose_config, nn.Module):
        pose_model = args.pose_config
    else:
        pose_model = init_model(args.pose_config, args.pose_checkpoint, args.device)
    
    return pose_model

def init_action_model(args):
    action_config = mmengine.Config.fromfile(args.action_config)
    action_model = init_recognizer(action_config,
                                   args.action_checkpoint,
                                   args.device)
    
    return action_model

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

def pose_inference(model,
                   frame_paths: List[str],
                   det_results: List[np.ndarray],
                   ) -> tuple:
    """Perform Top-Down pose estimation.

    Args:
        model (nn.Module): Initiated HRNet-w32 Model.
        frame_paths (List[str]): The paths of frames to do pose inference.
        det_results (List[np.ndarray]): List of detected human boxes.

    Returns:
        List[List[Dict[str, np.ndarray]]]: List of pose estimation results.
        List[:obj:`PoseDataSample`]: List of data samples, generally used
            to visualize data.
    """
    try:
        from mmpose.apis import inference_topdown
        from mmpose.structures import PoseDataSample, merge_data_samples
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`. These apis '
                          'are required in this inference api! ')

    results = []
    data_samples = []
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose_data_samples: List[PoseDataSample] \
            = inference_topdown(model, f, d[..., :4], bbox_format='xyxy')

        if d[..., :4] is None or len(d[..., :4]) == 0:
            for pds in pose_data_samples:
                pds.pred_instances.keypoints[:] = 0

        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = model.dataset_meta
        # make fake pred_instances
        if not hasattr(pose_data_sample, 'pred_instances'):
            num_keypoints = model.dataset_meta['num_keypoints']
            pred_instances_data = dict(
                keypoints=np.empty(shape=(0, num_keypoints, 2)),
                keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                bbox_scores=np.empty(shape=(0), dtype=np.float32))
            pose_data_sample.pred_instances = InstanceData(
                **pred_instances_data)

        poses = pose_data_sample.pred_instances.to_dict()
        results.append(poses)
        data_samples.append(pose_data_sample)

    return results, data_samples

def read_track_ann(fname=None, ):
    """Read ByteTrack Results.
    Track Results : [[frame_id, track_id, x, y, w, h, score, -1, -1, -1]].
    frame_id starts with 0, track_id starts with 1.

    Args:
        fname (str): ByteTrack inference results.

    Returns:
        List[List[np.ndarray]] : List of track results.
        len(List[List[np.ndarray]]) : num of people detected.
        len(List[np.ndarray]) : total frame.
        np.ndarray.shape : (1, 4) == (tlx, tly, tlx+w, tly+h)
    """
    assert fname is not None
    
    print(f'Reading {fname}')
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        lines[i] = list(map(float, line.split(',')))[:-1]
        lines[i][0] = int(lines[i][0]) # frame_id (0-indexed)
        lines[i][1] = int(lines[i][1]) - 1 # track_id starts with 1, change to 0-index (1-indexed)
    
    track_results = np.array(lines)

    total_frame = int(track_results[:, 0].max(axis=0)) + 1 # ByteTrack results starts with frame 0
    num_people = int(track_results[:, 1].max(axis=0)) + 1

    print(f'Annotated Total Frame : {total_frame}')
    print(f'Detected # of People : {num_people}')

    # Fill bboxes with zero if track_id is not detected.
    # List[ndarray(total_frame, 4)], len(List) = num_people
    track_id_bboxes = [np.zeros((total_frame, 4)) for _ in range(num_people)]

    for track_arr in lines:
        frame_id, track_id, tlx, tly, w, h, score = track_arr
        entry = np.array([tlx, tly, tlx+w, tly+h])
        track_id_bboxes[track_id][frame_id] = entry
    
    final_track_results = []
    for person_tracked in track_id_bboxes:
        sub_ls = [person_tracked[i:i+1,:] for i in range(person_tracked.shape[0])]
        final_track_results.append(sub_ls)
    
    return final_track_results

def extract_single_file(video_path, pose_model, out_root):
    # Get Tracking Results
    bytetrack_ann_path = video_path + '_related.txt'
    vid_id = video_path.split('/')[-1].split('.')[0]

    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames, fps = frame_extract(
        video_path, out_dir=tmp_dir.name
    )

    print(f'\nExtracting Pose from {video_path}')
    print(f'frame start : {frame_paths[0]}, frame end : {frame_paths[-1]}')
    print(f'len(frame_paths : {len(frame_paths)})')
    print(f'frame shape : {original_frames[0].shape}')
    print()

    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # List[List[np.ndarray]] : (# of person, # of frame, (1, 4))
    track_results = read_track_ann(bytetrack_ann_path)

    track_pose_results = []

    # Loop for each track_id
    for i, track_val in enumerate(track_results):
        # Get Pose Results
        # List[np.ndarray(1,4)]
        try:
            print(f'Performing Human Pose Estimation for track_id : {i} on each frame')
            pose_results, pose_datasample = pose_inference(
                pose_model,
                frame_paths,
                track_val
            )
            track_pose_results.append(pose_results)

            # align the num_person among frames
            print(type(pose_results))
            print(len(pose_results))
            print(pose_results[0]['keypoints'].shape)

            num_persons = max([pose_results[0]['keypoints'].shape[0] for pose in pose_results])
            num_points = pose_results[0]['keypoints'].shape[1]
            num_frames = len(pose_results)
            keypoints = np.zeros((num_persons, num_frame, num_points, 2), dtype=np.float32)
            scores = np.zeros((num_persons, num_frame, num_points), dtype=np.float32)

            for f_idx, frm_pose in enumerate(pose_results):
                frm_num_persons = frm_pose['keypoints'].shape[0]
                for p_idx in range(frm_num_persons):
                    keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
                    scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

            anno = dict()
            anno['keypoint'] = keypoints
            anno['keypoint_score'] = scores
            anno['frame_dir'] = vid_id
            anno['img_shape'] = (h, w)
            anno['original_shape'] = (h, w)
            anno['total_frames'] = keypoints.shape[1]
            anno['label'] = -1

            out_path = f'{os.path.join(out_root, vid_id)}_{i}.pkl'
            mmengine.dump(anno, out_path)

            # with open(f'{out_path}', 'wb') as handle:
            #     pickle.dump(track_pose_results, 
            #                 handle, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print('Error while inferencing HRNet-w32')
            print(f'{e}')
            tmp_dir.cleanup()
        
    tmp_dir.cleanup()

def extract_directory(dir_path, pose_model, out_root):
    video_ls = [os.path.join(dir_path, vpth) for vpth in os.listdir(dir_path) if vpth.endswith(('.mp4', '.avi'))]

    for video_path in video_ls:
        extract_single_file(video_path, pose_model, out_root)
    

def main():
    args, parser = parse_args()

    pose_model = init_pose_model(args)

    if args.file:
        if len(args.paths) != 1:
            parser.error('File mode requires exactly one file path.')

        extract_single_file(args.paths[0], pose_model, args.out_root)

    elif args.directory:
        if len(args.paths) != 1:
            parser.error('Single Directory mode requires exactly one directory path.')

        dir_path = args.paths[0]
        extract_directory(dir_path, pose_model, args.out_root)

    elif args.parent_dirs:
        print(args.paths)
        parent_dir = args.paths[0]

        if not os.path.exists(parent_dir):
            raise ValueError(f"경로가 존재하지 않습니다: {parent_dir}")
        
        dir_ls = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
                    if os.path.isdir(os.path.join(parent_dir, d))]

        for dir_path in tqdm(dir_ls):
            print('#' * 50)
            print(f'Extracting from {dir_path}')
            extract_directory(dir_path, pose_model, args.out_root)

    elif args.multiple_dirs:
        for dir_path in args.paths:
            extract_directory(dir_path, pose_model, args.out_root)



if __name__ == '__main__':
    main()