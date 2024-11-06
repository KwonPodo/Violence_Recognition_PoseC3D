import pickle
import os
from pprint import pprint

def read_pickle_file(file_path):
    """
    주어진 경로의 pickle 파일을 읽어서 내용을 반환합니다.
    
    Args:
        file_path (str): pickle 파일의 경로
        
    Returns:
        object: pickle 파일의 내용
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        pickle.UnpicklingError: pickle 파일을 읽는 데 실패했을 때
    """
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
    # 파일 확장자 확인
    if not file_path.endswith('.pkl'):
        print(f"경고: 파일 확장자가 .pkl이 아닙니다: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"pickle 파일을 읽는 데 실패했습니다: {str(e)}")
    except Exception as e:
        raise Exception(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")

def print_pickle_content(file_path, max_depth=None):
    """
    pickle 파일의 내용을 깔끔하게 출력합니다.
    
    Args:
        file_path (str): pickle 파일의 경로
        max_depth (int, optional): 출력할 최대 깊이. None이면 모두 출력
    """
    try:
        data = read_pickle_file(file_path)
        print(f"\n[{file_path}] 파일의 내용:")
        id0 = data[0]
        id0_frame0 = id0[0]
        print(id0_frame0)
        print()
        print(len(data))
        print(len(id0))
        print(id0_frame0['keypoints'].shape)
        print(id0_frame0['keypoint_scores'].shape)
        # pprint(data, depth=max_depth)
    except Exception as e:
        print(f"오류 발생: {str(e)}")

# 사용 예시
if __name__ == "__main__":
    # 단순 사용
    # file_path = "pipeline_integration/sample/long_subset/1_071_1_04.pkl"
    file_path = "custom_tools/sample/etri_sample/241101_0_out_45view_throw_walk0.pkl"
    print_pickle_content(file_path)
    
    # 출력 깊이 제한하여 사용
    # print_pickle_content(file_path, max_depth=2)