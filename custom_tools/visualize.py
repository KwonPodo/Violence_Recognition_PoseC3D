# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import tempfile
import warnings
import time
import os
from pprint import pprint

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.structures import InstanceData

from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract

try:
    from mmdet.apis import init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `init_detector` form `mmdet.apis`. '
                  'These apis are required in skeleton-based applications! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]


def visualize(args,
              frames,
              annotations,
              pose_data_samples,
              action_result,
              plate=PLATEBLUE,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.
        pose_data_samples (list[list[PoseDataSample]): The pose results.
        action_result (str): The predicted action recognition results.
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    frames_ = cp.deepcopy(frames)
    frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    # add pose results
    if pose_data_samples:
        pose_config = mmengine.Config.fromfile(args.pose_config)
        visualizer = VISUALIZERS.build(pose_config.visualizer)
        visualizer.set_dataset_meta(pose_data_samples[0].dataset_meta)
        for i, (d, f) in enumerate(zip(pose_data_samples, frames_)):
            visualizer.add_datasample(
                'result',
                f,
                data_sample=d,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show=False,
                wait_time=0,
                out_file=None,
                kpt_thr=0.3)
            frames_[i] = visualizer.get_image()
            cv2.putText(frames_[i], action_result, (10, 30), FONTFACE,
                        FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]

            # add action result for whole video
            cv2.putText(frame, action_result, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

            # add spatio-temporal action detection results
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_data_samples:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, f'{score[k]:.3f}'])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        # default=('work_dirs/Nov_3/custom_violence_keypoint_epoch300_batch8/best_acc_top1_epoch_264.pth'),
        # default=('work_dirs/train+val_uniform_cliplen30_epoch300_batch8/best_acc_top1_epoch_204.pth'),
        # default=('work_dirs/train+val_epoch_300_batch8/best_acc_top1_epoch_193.pth'),
        default=('work_dirs/uniform_cliplen48_epoch300_batch8/best_acc_top1_epoch_236.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='custom_tools/configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='custom_tools/configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/custom_violence_keypoint_epoch300_batch8.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='work_dirs/Nov_3/custom_violence_keypoint_epoch300_batch8/best_acc_top1_epoch_264.pth',
        # default='work_dirs/custom_violence_keypoint_epoch300_batch8/best_acc_top1_epoch_175.pth',
        # default='work_dirs/train+val_epoch_300_batch8/best_acc_top1_epoch_193.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.2,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='custom_tools/sample/etri_sample/241101_0_out_45view_throw_walk0.mp4',
        help='video file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='data/Final_dataset/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map',
        default='custom_tools/train_data/label_map.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-dir',
        # default='custom_tools/visualized',
        default='custom_tools/visualized/new_dataset',
        help='output dirname')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a spatio-temporal detection prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
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


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


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


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou


def skeleton_based_action_recognition(args, pose_results, h, w):
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset

    skeleton_model = init_recognizer(
        skeleton_config, args.skeleton_checkpoint, device=args.device)
    result = inference_skeleton(skeleton_model, pose_results, (h, w))
    action_idx = result.pred_score.argmax().item()
    return label_map[action_idx]


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w):
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config.model.cls_head.num_classes = num_class
    init_start_time = time.time()
    skeleton_stdet_model = init_recognizer(skeleton_config,
                                           args.skeleton_stdet_checkpoint,
                                           args.device)

    init_end_time = time.time()
    print(f'init time measured : {init_end_time - init_start_time}')
    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        pose_result = [pose_results[ind] for ind in frame_inds]

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            person_bbox = proposal[i][:4]
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses['keypoints']) == 0:
                    continue
                for k, bbox in enumerate(poses['bboxes']):
                    iou = cal_iou(bbox, area)
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses['keypoints'][index]
                keypoint_score[0, j] = poses['keypoint_scores'][index]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            output = inference_recognizer(skeleton_stdet_model, fake_anno)
            # for multi-label recognition
            score = output.pred_score.tolist()
            # print()
            # print('#'*20)
            # print(f'timestamps : {timestamps}')
            # print(f'timestamp : {timestamp}')
            # print(f'score : {score}')
            for k in range(len(score)):  # 81
                if k not in label_map:
                    continue
                if score[k] > args.action_score_thr:
                    skeleton_prediction[i].append((label_map[k], score[k]))
            print(f'skeleton_prediction : {skeleton_prediction}')
            print()

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    return timestamps, skeleton_predictions


def main():
    args = parse_args()
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames, fps = frame_extract(
        args.video, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results and pose results
    human_detections, _ = detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        device=args.device)
    pose_datasample = None
    pose_results, pose_datasample = pose_inference(
        args.pose_config,
        args.pose_checkpoint,
        frame_paths,
        human_detections,
        device=args.device)
    
    print(len(pose_results))
    print(pose_results[0])
    print(pose_results[0]['keypoints'].shape)

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Load spatio-temporal detection label_map
    stdet_label_map = load_label_map(args.label_map_stdet)

    action_result = None
    print('Use skeleton-based recognition')
    action_result = skeleton_based_action_recognition(
        args, pose_results, h, w)

    stdet_preds = None
    print('Use skeleton-based SpatioTemporal Action Detection')
    clip_len, frame_interval = 48, 1
    start_time = time.time()
    timestamps, stdet_preds = skeleton_based_stdet(args, stdet_label_map,
                                                    human_detections,
                                                    pose_results, num_frame,
                                                    clip_len,
                                                    frame_interval, h, w)
    end_time = time.time()

    print(f'stdet time measured : {end_time - start_time:.2f}')
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    # print(len(timestamps), len(stdet_preds))
    # print()
    # print(len(timestamps))
    # pprint(timestamps)
    # print()
    # print(len(stdet_preds))
    # pprint(stdet_preds)

    stdet_results = []
    for timestamp, prediction in zip(timestamps, stdet_preds):
        human_detection = human_detections[timestamp - 1]
        stdet_results.append(
            pack_result(human_detection, prediction, new_h, new_w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(args.predict_stepsize / args.output_stepsize)
    output_timestamps = dense_timestamps(timestamps, dense_n)
    frames = [
        cv2.imread(frame_paths[timestamp - 1])
        for timestamp in output_timestamps
    ]

    pose_datasample = [
        pose_datasample[timestamp - 1] for timestamp in output_timestamps
    ]

    vis_start = time.time()
    vis_frames = visualize(args, frames, stdet_results, pose_datasample,
                           action_result)
    vis_end = time.time()
    print(f'vis time : {vis_end - vis_start}')
    vid = mpy.ImageSequenceClip(vis_frames, fps=args.output_fps)

    vid_id, ext = os.path.splitext(os.path.basename(args.video))
    out_filename = os.path.join(args.out_dir, f'{vid_id}_{clip_len}{ext}')
    vid.write_videofile(out_filename)

    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
