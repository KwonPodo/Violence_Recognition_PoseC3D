import argparse
import time
import pickle
import os
from typing import List, Optional, Tuple
from tqdm import tqdm
from pprint import pprint

import torch
from torch import nn
import numpy as np
import mmengine
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmengine.dataset import Compose, pseudo_collate
from mmengine.utils import track_iter_progress

from mmaction.apis import init_recognizer
from mmaction.structures import ActionDataSample

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
        default='pipeline_build/demo_configs/violence_custom_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py',
        help='skeleton-based action recognition config file path'
    )
    parser.add_argument(
        '--action-checkpoint',
        # default='pipeline_build/weights/epoch_72.pth',
        default='pipeline_build/weights/best_acc_top1_epoch_62.pth',
        help='skeleton-based action recognition checkpoint file/url'
    )
    parser.add_argument(
        '--pose-file-path',
        default='pipeline_build/sample/long_subset/1_071_1_04.pkl',
        help='Path of saved pose data'
    )
    parser.add_argument(
        '--action-save-dir',
        default='pipeline_build/sample/results',
        help='Directory path to save action results'
    )
    parser.add_argument(
        '--label-map',
        default='custom_tools/label_map.txt',
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
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option'
    )

    args = parser.parse_args()
    return args


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

def preprocess_pose_whole_video(model: nn.Module,
                         pose_results: List[dict],
                         img_shape: Tuple[int],
                         test_pipeline: Optional[Compose] = None
                         ) -> dict:
    """Preprocess pose results for the action recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        pose_results (List[dict]): The pose estimation results dictionary
            (the results of `pose_inference`) of length T.
        img_shape (Tuple[int]): The original image shape used for inference
            skeleton recognizer.
        test_pipeline (:obj:`Compose`, optional): The test pipeline.
            If not specified, the test pipeline in the config will be
            used. Defaults to None.

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
    num_frame = len(pose_results)
    num_person = max([len(x['keypoints']) for x in pose_results])
    dummy_input = dict(
        frame_dict='',
        label=-1,
        img_shape=(h, w),
        origin_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame
    )

    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                              dtype=np.float16)
    
    for frame_idx, frame_pose in enumerate(pose_results):
        frame_num_persons = frame_pose['keypoints'].shape[0]
        for p_idx in range(frame_num_persons):
            keypoint[frame_idx, p_idx] = frame_pose['keypoints'][p_idx]
            keypoint_score[frame_idx, p_idx] = frame_pose['keypoint_scores'][p_idx]

    # [num_person==1, num_frame, num_keypoints, 2]
    dummy_input['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    dummy_input['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    pose_heatmap = test_pipeline(dummy_input)
    pose_heatmap = pseudo_collate([pose_heatmap])

    return pose_heatmap

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

def main():
    args = parse_args()

    # Get Pose Results
    with open(args.pose_file_path, 'rb') as handle:
        pose_results = pickle.load(handle)
    
    action_model = init_action_model(args)
    img_shape = (256, 192)

    print(f'Total frame : {len(pose_results[0])}')
    print(f'Number of person detected : {len(pose_results)}')

    track_id = 0
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
            track_id_action_result.append(
                {
                    'pred_score': action_result.pred_score.cpu().numpy(),
                    'pred_label': action_result.pred_label.cpu().numpy(),
                    'frame_index': frame_inds
                }
            )
            prog_bar.update()
        end_time = time.time()
        inference_time = end_time - start_time

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
        pickle.dump(track_id_action_result, handle)


if __name__ == '__main__':
    main()