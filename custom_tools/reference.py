import cv2
import numpy as np
from pathlib import Path

def visualize_video(
    video_path,
    pose_results,
    predictions,
    output_path,
    keypoint_info,
    skeleton_info,
    window_size=48,
    stride=12,
    conf_threshold=0.5
):
    """
    비디오에 action classification 결과와 COCO pose keypoint를 시각화합니다.
    
    Args:
        video_path (str): 입력 비디오 경로
        pose_results (dict): 프레임별 keypoint 결과
        predictions (list): [{'frame_index': int, 'pred_label': str, 'pred_score': float}, ...]
        output_path (str): 출력 비디오 저장 경로
        keypoint_info (dict): COCO keypoint 정보
        skeleton_info (dict): COCO skeleton 정보
        window_size (int): sliding window 크기
        stride (int): window 이동 간격
        conf_threshold (float): confidence threshold
    """
    # 비디오 읽기
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 비디오 writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 프레임별 예측 결과 매핑
    frame_to_prediction = {}
    for pred in predictions:
        start_frame = pred['frame_index']
        # 해당 window의 모든 프레임에 예측 결과 매핑
        for frame in range(start_frame, min(start_frame + window_size, total_frames)):
            if frame not in frame_to_prediction or pred['pred_score'] > frame_to_prediction[frame]['pred_score']:
                frame_to_prediction[frame] = pred

    def draw_skeleton(frame, keypoints, skeleton_info, keypoint_info):
        """프레임에 skeleton을 그립니다."""
        # Keypoints 그리기
        for kid in range(len(keypoints)):
            x, y, conf = keypoints[kid]
            if conf > 0:  # keypoint가 검출된 경우
                color = keypoint_info[kid]['color']
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        
        # Skeleton 그리기
        for sk_id, sk in skeleton_info.items():
            pos1 = None
            pos2 = None
            
            # skeleton의 시작점과 끝점 찾기
            for kid, kpt_info in keypoint_info.items():
                if kpt_info['name'] == sk['link'][0]:
                    pos1 = keypoints[kid][:2]
                if kpt_info['name'] == sk['link'][1]:
                    pos2 = keypoints[kid][:2]
            
            # 두 점이 모두 검출된 경우에만 선 그리기
            if pos1 is not None and pos2 is not None:
                if keypoints[list(keypoint_info.keys())[0]][2] > 0 and keypoints[list(keypoint_info.keys())[1]][2] > 0:
                    cv2.line(frame, (int(pos1[0]), int(pos1[1])),
                            (int(pos2[0]), int(pos2[1])), sk['color'], 2)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 현재 프레임의 예측 결과 표시
        if frame_idx in frame_to_prediction:
            pred = frame_to_prediction[frame_idx]
            if pred['pred_score'] >= conf_threshold:
                # Window 정보 표시
                window_text = f"Window {pred['frame_index']}-{pred['frame_index'] + window_size}"
                cv2.putText(frame, window_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # Action 예측 결과 표시
                pred_text = f"{pred['pred_label']}: {pred['pred_score']:.2f}"
                cv2.putText(frame, pred_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
        
        # Pose 시각화
        if frame_idx in pose_results:
            keypoints = pose_results[frame_idx]
            draw_skeleton(frame, keypoints, skeleton_info, keypoint_info)
        
        # 현재 프레임 번호 표시
        cv2.putText(frame, f"Frame: {frame_idx}", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 프레임 저장
        out.write(frame)
        frame_idx += 1
    
    # 리소스 해제
    cap.release()
    out.release()

# 사용 예시:
"""
# Args 예시
args = {
    'video_path': 'input.mp4',
    'output_path': 'output.mp4',
    'window_size': 48,
    'stride': 12,
    'conf_threshold': 0.5
}

# 실행
visualize_video(
    video_path=args['video_path'],
    pose_results=pose_results,
    predictions=predictions,
    output_path=args['output_path'],
    keypoint_info=keypoint_info,
    skeleton_info=skeleton_info,
    window_size=args['window_size'],
    stride=args['stride'],
    conf_threshold=args['conf_threshold']
)
"""