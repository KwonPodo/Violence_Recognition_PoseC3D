import pickle
import os
from pathlib import Path
from collections import Counter
from pprint import pprint

def update_selected_pickles(source_dir, txt_path, output_dir):
    """
    특정 디렉토리의 pickle 파일들 중 txt 파일에 명시된 파일들만
    label을 수정하여 새로운 디렉토리에 저장하는 함수
    
    Args:
        source_dir (str): pickle 파일들이 있는 원본 디렉토리 경로
        txt_path (str): pickle_name과 label이 있는 txt 파일 경로
        output_dir (str): 수정된 pickle 파일들을 저장할 디렉토리 경로
    """
    label_map = {
        0: 'throw',
        1: 'protest',
        2: 'raise',
        3: 'robbery',
        4: 'fight',
        5: 'fall',
        6: 'assault',
        7: 'vandalism',
        8: 'trespass',
        9: 'stabbing'
    }
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # txt 파일에서 파일명과 label 읽기
    filename_labels = {}
    selected_filename_ls = []
    non_exist_ls = []
    labels = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            filename, label = [x.strip() for x in line.split(',')]
            filename_with_ext = filename + '.pkl'
            filename_labels[filename_with_ext] = int(label)
            selected_filename_ls.append(filename_with_ext)
            labels.append(int(label))
    
    # 레이블 통계 출력
    counter = Counter(sorted(labels))
    for k in counter.keys():
        print(f'{label_map[k]} : {counter[k]}')
    print(f'Total Annotated Count : {sum(counter.values())}\n')
    
    # 소스 디렉토리의 모든 파일 목록을 미리 생성
    source_files = {}
    for source_file in Path(source_dir).rglob('*.pkl'):
        source_files[source_file.name] = source_file
    
    # 처리 카운터 초기화
    processed_count = 0
    skipped_count = 0
    
    # 선택된 파일들 처리
    for filename in selected_filename_ls:
        if filename not in source_files:
            print(f"Missing file: {filename}")
            non_exist_ls.append(filename)
            continue
            
        source_file = source_files[filename]
        
        try:
            # pickle 파일 읽기
            with open(source_file, 'rb') as f:
                data = pickle.load(f)
            
            # label 업데이트
            data['label'] = filename_labels[filename]
            
            # 새로운 경로에 저장
            output_path = Path(output_dir) / filename
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
                
            processed_count += 1
            print(f"Success: {filename}의 label이 {filename_labels[filename]}로 업데이트되어 저장되었습니다.")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            skipped_count += 1
    
    print(f"\n처리 완료:")
    print(f"- 수정되어 저장된 파일: {processed_count}개")
    print(f"- 건너뛴 파일: {skipped_count}개")
    print(f"- 저장 위치: {output_dir}")
    print(f'\nNon Existing files that are in the txt file but not in the source dir:')
    pprint(non_exist_ls)

# 실행 예시
if __name__ == "__main__":
    # 경로 설정
    source_directory = "custom_tools/train_data/pose_pkls"  # pickle 파일들이 있는 원본 디렉토리
    labels_txt = "custom_tools/train_data/pkl_tools/select_track_id_cls.txt"  # pickle_name과 label이 있는 txt 파일
    output_directory = "custom_tools/train_data/selected_pose_pkls"  # 수정된 파일들을 저장할 디렉토리
    
    update_selected_pickles(source_directory, labels_txt, output_directory)