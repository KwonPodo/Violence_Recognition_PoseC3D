import argparse
import time
import pickle
import os
from typing import List, Optional, Tuple
from tqdm import tqdm
from collections import deque
from pprint import pprint

import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import copy as cp
import moviepy.editor as mpy

import mmengine
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmengine.dataset import Compose, pseudo_collate
from mmengine.utils import track_iter_progress

from mmaction.apis import init_recognizer
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract

class ActionVoter:
    def __init__(
        self,
        buffer_size : int = 3,
        min_votes : int = 3,
        method : str ='hard'
    ):
    
        self.buffer_size = buffer_size
        self.method = method
        self.min_votes = min_votes

        self.pred_history = deque(maxlen=buffer_size)
        self.score_history = deque(maxlen=buffer_size)

        self.last_pred = None
    
    def vote(self, pred_info:dict) -> dict:
        if self.method == 'off':
            return pred_info
        
        current_pred = self._to_numpy(pred_info.pred_label[0])
        current_scores = self._to_numpy(pred_info.pred_score)

        self.pred_history.append(current_pred)
        self.score_history.append(current_scores)

        if len(self.pred_history) < self.min_votes:
            return pred_info
        
        if self.method == 'hard':
            voted_label = self._hard_vote()
        elif self.method == 'soft':
            voted_label = self._soft_vote()
        else:
            raise ValueError(f'Unknown voting method is given : {self.voting_method}')
        
        result = cp.deepcopy(pred_info)
        
        if isinstance(pred_info.pred_label, torch.Tensor):
            device = pred_info.pred_label.device
            voted_label = torch.tensor([voted_label], device=device)
        else:
            voted_label = np.array([voted_label])

        result.pred_label = voted_label

        self.last_pred = voted_label

        return result
    
    def _to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        
        return data

    def _hard_vote(self) -> int:
        predictions = list(self.pred_history)
        unique_labels, counts = np.unique(predictions, return_counts=True)

        return unique_labels[np.argmax(counts)]
    
    def _soft_vote(self) -> int:
        print()
        print(self.score_history)
        print(list(self.score_history))
        print(np.mean(list(self.score_history), axis=0))
        print()

        avg_scores = np.mean(list(self.score_history), axis=0)

        return np.argmax(avg_scores)

    def set_method(self, method:str):
        valid_methods = {'soft', 'hard', 'off'}
        if method not in valid_methods:
            raise ValueError(f'Method must be one of {valid_methods}')

        self.method = method
        

def parse_args():
    parser = argparse.ArgumentParser(description='Violence Action Recognition Demo')
    parser.add_argument(
        '--pose-config',
        default='pipeline_build/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='HRNet Config file path (from mmpose)'
    )
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='HRNet-w32 Checkpoint file/url'
    )
    parser.add_argument(
        '--action-config',
        default='pipeline_integration/demo_configs/custom_violence_keypoint_epoch300_batch8.py',
        help='skeleton-based action recognition config file path'
    )
    parser.add_argument(
        '--action-checkpoint',
        default='pipeline_integration/weights/best_acc_top1_epoch_193.pth',
        help='skeleton-based action recognition checkpoint file/url'
    )
    parser.add_argument(
        '--pose-file-path',
        default='custom_tools/sample/etri_sample/241101_0_out_45view_throw_walk0.pkl',
        help='Path of saved pose data'
    )
    parser.add_argument(
        '--action-save-dir',
        default='custom_tools/sample/results',
        help='Directory path to save action results'
    )
    parser.add_argument(
        '--label-map',
        default='pipeline_integration/label_map.txt',
        help='label map file for action recognition'
    )
    parser.add_argument(
        '--clip-len',
        default=48,
        help='number of frames in a clip'
    )
    parser.add_argument(
        '--predict-step-size',
        default=12,
        help='step size of the sliding window'
    )
    parser.add_argument(
        '--use-vote',
        action='store_true',
        help='flag for using vote'
    )
    parser.add_argument(
        '--vote-method',
        type=str,
        default='hard'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='visualize flag.'
    )
    parser.add_argument(
        '--video-path',
        default='custom_tools/sample/etri_sample/241101_0_out_45view_throw_walk0.mp4',
        help='video path for visualization. Only used when visualize flag is True.'
    )
    parser.add_argument(
        '--video-out',
        default='custom_tools/sample/results/out.mp4',
        help='output path for visualized video. Only used when visualize flag is True.'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option'
    )

    args = parser.parse_args()
    return args

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

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
    print('Performing Human Pose Estimation for each frame')
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

def preprocess_pose_st(model: nn.Module,
                         pose_results: List[dict],
                         img_shape: Tuple[int],
                         test_pipeline: Optional[Compose] = None,
                         frame_interval: int=1,
                         clip_len: int=48,
                         predict_step_size: int=12
                         ) -> dict:
    """Preprocess pose results spatio-temporally for the action recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        pose_results (List[dict]): The pose estimation results dictionary
            (the results of `pose_inference`) of length T.
        img_shape (Tuple[int]): The original image shape used for inference
            skeleton recognizer.
        test_pipeline (:obj:`Compose`, optional): The test pipeline.
            If not specified, the test pipeline in the config will be
            used. Defaults to None.
        clip_len (int): Number of frames segmented to inference into the model.
        frame_interval (int): Frame intervals within the clip.
        predict_step_size: (int): Step size of the sliding window.
            predict_step_size에 따라 sliding window가 더 큰 폭으로 이동하며, 전체 동영상의 inference time에 영향을 미침.

    Returns:
        pose_heatmap (dict): Pose Keypoints constructed into heatmap with UniformSampling.
            pose_heatmap['inputs'] (torch.Tensor): Constructed Heatmap.
            pose_heatmap['data_samples'] (ActionDataSample): Data structure for action results.
    """
    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    h, w = img_shape
    num_keypoint = pose_results[0]['keypoints'].shape[1]
    total_frame = len(pose_results)
    num_person = max([len(x['keypoints']) for x in pose_results])


    # Get Temporal timestamps
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'clip_len should be even'
    timestamps = np.arange(window_size // 2, total_frame + 1 - window_size // 2, predict_step_size)

    clip_pose_heatmaps = []
    clip_frame_inds = []

    for timestamp in timestamps:
        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_clip_frame = len(frame_inds)

        # print(f'start frame : {start_frame}')
        # print(f'frame_inds : {frame_inds}')
        # print(f'num_clip_frame : {num_clip_frame}')
        # print()

        clip_pose_results = [pose_results[idx] for idx in frame_inds]

        dummy_input = dict(
            frame_dict='',
            label=-1,
            img_shape=(h, w),
            origin_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=num_clip_frame
        )

        keypoint = np.zeros((num_clip_frame, num_person, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_clip_frame, num_person, num_keypoint),
                                dtype=np.float16)
        
        for frame_idx, frame_pose in enumerate(clip_pose_results):
            keypoint[frame_idx, 0] = frame_pose['keypoints'][0]
            keypoint_score[frame_idx, 0] = frame_pose['keypoint_scores'][0]
            # for p_idx in range(frame_num_persons):
            #     keypoint[frame_idx, p_idx] = frame_pose['keypoints'][p_idx]
            #     keypoint_score[frame_idx, p_idx] = frame_pose['keypoint_scores'][p_idx]

        dummy_input['keypoint'] = keypoint.transpose((1, 0, 2, 3)) # [num_person==1, num_frame, num_keypoints, 2]
        dummy_input['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

        pose_heatmap = test_pipeline(dummy_input)
        pose_heatmap = pseudo_collate([pose_heatmap])

        clip_pose_heatmaps.append(pose_heatmap)
        clip_frame_inds.append(frame_inds)

    return clip_pose_heatmaps, clip_frame_inds

def action_inference(model: nn.Module,
                     pose_result: dict,
                     ) -> ActionDataSample:
    """Inference pose results with the action recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        pose_results (List[dict]): The pose estimation results dictionary
            (the results of `pose_inference`) of length T.

    Returns:
        :obj:`ActionDataSample`: The inference results. Specifically, the
        predicted scores are saved at ``result.pred_score``.
    """

    with torch.no_grad():
        result = model.test_step(pose_result)[0]
    
    return result

def postprocess_clip_results(result, args):
    label_map = load_label_map(args.label_map)
    num_person = len(result)
    timestamps = [clip_info['frame_index'][0] for clip_info in result[0]]
    print(timestamps)
    print(len(timestamps))

    clip_zip = list(zip(*result))

    timestamps = []
    predictions = []
    for clip in clip_zip:
        prediction = []
        for i, person_clip_info in enumerate(clip):
            prediction.append([])
            timestamp = person_clip_info['frame_index'][0]
            pred_label_id = person_clip_info['pred_label'][0]
            pred_label_str = label_map[pred_label_id]
            pred_label_score = person_clip_info['pred_score'][pred_label_id]
            timestamps.append(timestamp)
            prediction[i].append((pred_label_str, pred_label_score))

        predictions.append(prediction)

    return timestamps, predictions

def visualize(pose_results, timestamps, predictions, args):
    # COCO keypoints pairs for drawing lines between keypoints
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ]

    keypoint_info = keypoint_info[0]
    skeleton_info = skeleton_info[0]
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames, fps = frame_extract(
        args.video_path, out_dir=tmp_dir.name
    )
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Map prediction results per frame
    
    def draw_skeleton(frame, keypoints, keypoint_scores, skeleton_info, keypoint_info):
        """Draw Skeleton on a frame."""

        # draw keypoints
        for k_id in range(len(keypoints)):
            x, y = keypoints[k_id]
            conf = keypoint_scores[k_id]
            if conf > 0:
                color = keypoint_info[k_id]['color']
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        # draw skeletons
        for sk_id, sk in skeleton_info.items():
            pos1 = None
            pos2 = None

            # get start point and end point of skeleton
            for k_id, kpt_info in keypoint_info.items():
                if kpt_info['name'] == sk['link'][0]:
                    pos1 = keypoints[k_id][:2]
                if kpt_info['name'] == sk['link'][1]:
                    pos2 = keypoints[k_id][:2]
            
            if pos1 is not None and pos2 is not None:
                cv2.line(frame, (int(pos1[0]), int(pos1[1])),
                            (int(pos2[0]), int(pos2[1])), sk['color'], 2)

        return frame_
    

    vis_frames = []
    for fpth in frame_paths:
        frame = cv2.imread(fpth)[...,::-1]
        frame_ = cp.deepcopy(frame)
        vis_frames.append(frame_)

    print(len(pose_results))
    for single_person_pose_result in pose_results:
        i = 0
        for frame2vis, pose_res in zip(vis_frames, single_person_pose_result):
            keypoints = pose_res['keypoints'][0]
            keypoint_scores = pose_res['keypoint_scores'][0]
            vis_frame = draw_skeleton(frame2vis, keypoints, keypoint_scores, skeleton_info, keypoint_info)
            vis_frames[i] = vis_frame
    
    video = mpy.ImageSequenceClip(vis_frames, fps=fps)
    video.write_videofile(args.video_out)

    tmp_dir.cleanup()



def main():
    args = parse_args()

    # Get Pose Results
    with open(args.pose_file_path, 'rb') as handle:
        pose_results = pickle.load(handle)
    
    action_model = init_action_model(args)
    img_shape = (256, 192)

    print(f'Total frame : {len(pose_results[0])}')
    print(f'Number of person detected : {len(pose_results)}')

    action_voter = ActionVoter(
        buffer_size=3,
        min_votes=3,
        method=args.vote_method
    )

    track_id = 0
    result = []
    for track_id_pose in tqdm(pose_results):
        print(f'\n\nProcessing track_id : {track_id}')
        print(f'Preprocessing pose data of {track_id} into sliding window clips...')
        track_id += 1
        start_time = time.time()
        clip_pose_heatmaps, clip_frame_inds = preprocess_pose_st(action_model, 
                                                                 track_id_pose, img_shape, 
                                                                 clip_len=args.clip_len, 
                                                                 predict_step_size=args.predict_step_size)
        end_time = time.time()

        process_time = end_time - start_time
        print(f'Preprocess time : {process_time:.2f}s')

        print('Inferencing Clips into Action Recognition Model')
        track_id_action_result = []
        prog_bar = mmengine.ProgressBar(len(clip_pose_heatmaps))

        start_time = time.time()
        for pose_heatmaps, frame_inds in list(zip(clip_pose_heatmaps, clip_frame_inds)):
            action_result = action_inference(action_model, pose_heatmaps)
            voted_action_result = action_voter.vote(action_result)

            track_id_action_result.append(
                {
                    'pred_score': voted_action_result.pred_score.cpu().numpy(),
                    'pred_label': voted_action_result.pred_label.cpu().numpy(),
                    'frame_index': frame_inds
                }
            )

            # track_id_action_result.append(
            #     {
            #         'pred_score': action_result.pred_score.cpu().numpy(),
            #         'pred_label': action_result.pred_label.cpu().numpy(),
            #         'frame_index': frame_inds
            #     }
            # )

            prog_bar.update()
        
        end_time = time.time()
        inference_time = end_time - start_time
        result.append(track_id_action_result)

        print(f'\nInference time : {inference_time:.2f}s\tFPS : {len(clip_frame_inds) * args.clip_len / inference_time}')
    

    '''
    track_id_action_result : List[dict]: List with length equal to number of sliding window shifts.
    track_id_action_result[num_shifts]['pred_score']: Contains predicted probability of the sliding window clip input
    track_id_action_result[num_shifts]['pred_label']: Contains predicted label of the sliding window clip input
    Corresponding labels are annotated at `pipeline_build/label_map.txt`
    '''
    vid_id = args.pose_file_path.split('/')[-1].split('.')[0]
    save_pth = os.path.join(args.action_save_dir, f'action_{vid_id}.pkl')
    with open(save_pth, 'wb') as handle:
        pickle.dump(result, handle)

    print(result)
    if args.visualize:
        timestamps, predictions = postprocess_clip_results(result, args)
        print(timestamps)
        print(predictions)

        visualize(pose_results, timestamps, predictions, args)


if __name__ == '__main__':
    main()