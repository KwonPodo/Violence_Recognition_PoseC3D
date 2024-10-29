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

from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
from mmaction.structures import ActionDataSample


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

def skeleton_based_action_recognition(args, pose_results, h, w):
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    # skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset

    skeleton_model = init_recognizer(
        skeleton_config, args.skeleton_checkpoint, device=args.device)
    result = inference_skeleton(skeleton_model, pose_results, (h, w))
    action_idx = result.pred_score.argmax().item()
    return label_map[action_idx]



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
        '--action-save-dir',
        default='pipeline_build/sample/results',
        help='Directory path to save action results'
    )
    parser.add_argument(
        '--label-map',
        default='custom_tools/train_data/label_map.txt',
        help='label map file for action recognition'
    )
    parser.add_argument(
        '--clip-len',
        default=48,
        help='number of frames in a clip'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Get Pose Results
    with open(args.pose_file_path, 'rb') as handle:
        pose_results = pickle.load(handle)
    
    action_model = init_action_model(args)
    img_shape = (256, 192)

    print(f'Total frame : {len(pose_results[0])}')
    print(f'Number of person detected : {len(pose_results)}')



if __name__ == '__main__':
    main()