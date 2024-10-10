import multiprocessing as mp
from multiprocessing import freeze_support
import json
import argparse
import os
import cv2
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe
from face_exp import mediapipe_inference, get_rectsize, swin_face

# Producer function
def producer(queue, video_path, total_frame):
    face_detection = mediapipe.solutions.face_detection
    face_detection = face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(video_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    while True:
        ret, frame = cap.read()
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if current_frame % 30 == 0:
            print(f"Process {mp.current_process().name} processing frame {current_frame}")

        if not ret:
            break
        else:
            coords = mediapipe_inference(frame, face_detection)
            if coords is not None:
                face_roi = frame[coords[2]:coords[3], coords[0]:coords[1]]
                face_roi = cv2.resize(face_roi, (224, 224))
                face_roi = transform(face_roi).cpu().numpy()  # Convert to NumPy array before putting it in the queue
                queue.put([current_frame, face_roi])
                last_roi = face_roi
            else:
                queue.put([current_frame, last_roi])
        
        if current_frame >= total_frame:
            break

# Consumer function
def consumer(queue, result_list, weights, device_num):
    device = torch.device("cuda:{}".format(device_num)) if torch.cuda.is_available() else torch.device("cpu")
    model = swin_face()
    checkpoint = weights
    model = nn.DataParallel(model)
    
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        swin_face_model = model.module
        swin_face_model = swin_face_model.eval().to(device)
        print(f"Process {mp.current_process().name} loaded model")

        frame_batch = []
        missing_frames = []

        while True:
            item = queue.get()

            # Sentinel 값인지 확인
            if item == 'LAST':
                print(f"Process {mp.current_process().name} received sentinel value")
                
                # 남아있는 배치가 있을 경우 처리
                if frame_batch:
                    print(f"Process {mp.current_process().name} processing remaining frames")
                    frame_batch = torch.stack(frame_batch).to(device)
                    result = swin_face_model(frame_batch)

                    # 계산 그래프 분리
                    result = result.detach().cpu()

                    # 결과를 result_list에 저장
                    for i, res in enumerate(result):
                        result_list[int(current_frame - len(frame_batch) + 1 + i)] = res

                    # 누락된 프레임 처리
                    # for frame in missing_frames:
                    #     result_list[int(frame)] = None
                break

            # 정상적인 데이터인 경우
            current_frame, face_roi = item

            # 프레임 인덱스가 결과 리스트 길이를 넘는 경우 종료
            if current_frame >= len(result_list):
                break

            # face_roi가 None인 경우 처리
            if face_roi is None:
                missing_frames.append(current_frame)
                frame_batch.append(torch.zeros((3, 224, 224)))
            else:
                face_roi = torch.from_numpy(face_roi)  # NumPy 배열을 다시 텐서로 변환
                frame_batch.append(face_roi)
            
            # 배치 크기가 30이 되었을 때 처리
            if len(frame_batch) == 30:
                print(f"Process {mp.current_process().name} processing frames {current_frame - 29} - {current_frame}")
                frame_batch = torch.stack(frame_batch).to(device)
                result = swin_face_model(frame_batch)

                # 계산 그래프 분리
                result = result.detach().cpu()

                # 결과를 result_list에 저장
                for i, res in enumerate(result):
                    result_list[int(current_frame - 30 + i)] = res
                
                # 다음 배치를 위해 frame_batch 및 missing_frames 초기화
                frame_batch = []
                missing_frames = []




# Find maximum probability channel function
def find_max_prob_channel(frame_result):
    max_prob = -1
    max_prob_channel = None
    
    for key, value in frame_result.items():
        prob = value
        if prob is not None and isinstance(prob, list):
            prob_last = prob[-1]
            if prob_last > max_prob:
                max_prob = prob_last
                max_prob_channel = key
    
    return max_prob_channel

from collections import Counter

def add_most_common_channel_per_interval(result_dict, switching_interval):
    total_frames = len(result_dict)

    for start_frame in range(0, total_frames, switching_interval):
        # 구간 내의 끝 프레임 계산
        end_frame = min(start_frame + switching_interval, total_frames)

        # 구간 내 max_prob_channel 값 수집
        channels_in_interval = []
        for frame in range(start_frame, end_frame):
            frame_str = str(frame)
            if frame_str in result_dict and "max_prob_channel" in result_dict[frame_str]:
                channel = result_dict[frame_str]["max_prob_channel"]
                if channel:
                    channels_in_interval.append(channel)

        # 가장 자주 등장한 채널 찾기
        if channels_in_interval:
            most_common_channel = Counter(channels_in_interval).most_common(1)[0][0]
        else:
            most_common_channel = None

        # 해당 구간의 모든 프레임에 most_common_channel_per_interval 추가
        for frame in range(start_frame, end_frame):
            frame_str = str(frame)
            if frame_str in result_dict:
                result_dict[frame_str]["most_common_channel_per_interval"] = most_common_channel

    return result_dict

@dataclass
class Speaker:
    name: str
    video_path: str
    queue: mp.Queue
    result_list: list
    producer: mp.Process = None
    consumer: mp.Process = None

if __name__ == "__main__":
    freeze_support()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--widechannel_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/W.mp4',
                        help='widechannel_video')
    
    parser.add_argument('--speaker_videos', type=str, nargs='+',
                    default=[
                        '/NasData/home/ych/Multicam_materials/thelive/C.mp4', 
                        '/NasData/home/ych/Multicam_materials/thelive/D.mp4', 
                        '/NasData/home/ych/Multicam_materials/thelive/MC_left.mp4', 
                        '/NasData/home/ych/Multicam_materials/thelive/MC_right.mp4'
                    ],
                    help='List of speaker videos')

    parser.add_argument('--weights', type=str, default='/NasData/home/ych/multi-channel-video-compile-2024/speaking_detection_model_weight.pth')
    args = parser.parse_args()

    run_start = time.time()

    # Getting total frames of the wide channel video
    tmp_cap = cv2.VideoCapture(args.widechannel_video)
    total_frame = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp_cap.release()

    # Video paths and speaker names
    spkr_video_paths = args.speaker_videos
    speaker_names = [s.split('/')[-1] for s in spkr_video_paths]
    
    # Creating speaker instances and multiprocessing lists and queues
    manager = mp.Manager()
    speakers = [
        Speaker(
            name=name,
            video_path=video_path,
            queue=mp.Queue(),
            result_list=manager.list([None for _ in range(total_frame)])
        )
        for name, video_path in zip(speaker_names, spkr_video_paths)
    ]

    # Start producer processes
    for speaker in speakers:
        speaker.producer = mp.Process(target=producer, args=(speaker.queue, speaker.video_path, total_frame))
        speaker.producer.start()
        print(f"{speaker.name} producer started")

    # Start consumer processes
    for i, speaker in enumerate(speakers):
        speaker.consumer = mp.Process(target=consumer, args=(speaker.queue, speaker.result_list, args.weights, i))
        speaker.consumer.start()
        print(f"{speaker.name} consumer started")

    # Wait for producer processes to finish
    for speaker in speakers:
        speaker.producer.join()
        print(f"{speaker.name} producer joined")
    
    # Send sentinel value to each consumer process to signal the end
    for speaker in speakers:
        speaker.queue.put('LAST')
        print(f"{speaker.name} sentinel value sent")

    # Wait for consumer processes to finish
    for speaker in speakers:
        speaker.consumer.join()
        print(f"{speaker.name} consumer joined")

    print(f"Elapsed time: {time.time() - run_start:.2f} sec")

    # Collect results and write to a JSON file
    result_dict = {}
    for i in range(total_frame):
        frame_result = {}
        for speaker in speakers:
            entry = speaker.result_list[i]
            if isinstance(entry, torch.Tensor):
                frame_result[speaker.name] = entry.tolist()  # 텐서를 리스트로 변환
            else:
                frame_result[speaker.name] = entry  # None 또는 다른 타입의 경우 그대로 유지
        frame_result["max_prob_channel"] = find_max_prob_channel(frame_result)
        result_dict[int(i)] = frame_result

    # max_prob_channel adjusting for every 30 frames with most common channel
    
    

    # 함수 실행 예시
    # switching_interval = 15  # 또는 30으로 설정 가능
    # result_dict_with_common_channel = add_most_common_channel_per_interval(result_dict, switching_interval)

    # # 결과 확인
    # import pprint
    # pprint.pprint(result_dict_with_common_channel)
    



    # JSON 파일로 저장
    with open("./multi_channel_lip_infer_exp1.json", 'w') as f:
        json.dump(result_dict, f, indent=4)

    # with open("./result_dict_with_common_channel.json", 'w') as f:
    #     json.dump(result_dict_with_common_channel, f, indent=4)
