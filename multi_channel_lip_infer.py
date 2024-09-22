import json
import argparse
import multiprocessing as mp
import cv2
from dataclasses import dataclass
from utils import lip_detection_in_video, infer_lip_state

def time_to_frames(time_in_seconds, frames_per_second=30):
    return int(time_in_seconds * frames_per_second)

def parse_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        trans_list = json.load(file)

    segments = {}

    for i, item in enumerate(trans_list):
        start_time = float(item['start'])
        end_time = float(item['end'])
        start_frame = time_to_frames(start_time)
        end_frame = time_to_frames(end_time)

        speaker = str(item['speaker'])

        segments[i] = {'start': start_time, 
                       'end': end_time,
                       'start_frame': start_frame,
                       'end_frame': end_frame, 
                       'speaker': speaker}
    
    return segments

def find_max_prob_channel(frame_result):
    # prob[-1]이 가장 큰 값을 가진 key를 찾기
    max_prob = -1
    max_prob_channel = None
    
    for key, value in frame_result.items():
        # prob이 None이 아닌지 확인하고, prob 리스트의 마지막 값 확인
        prob = value.get("prob")
        if prob != "None":
            prob_last = prob[-1]  # prob[-1]을 가져옴
            if prob_last > max_prob:
                max_prob = prob_last
                max_prob_channel = key
    
    return max_prob_channel

@dataclass
class Speaker:
    name: str
    video_path: str
    queue: mp.Queue
    result_list: mp.Manager().list
    producer: mp.Process = None
    consumer: mp.Process = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--widechannel_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/W.mp4',
                        help='widechannel_video')
    parser.add_argument('--speaker0_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/C.mp4', # C
                        help='speaker1_video')
    parser.add_argument('--speaker1_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/D.mp4', # D
                        help='speaker2_video')
    parser.add_argument('--speaker2_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/MC_left.mp4', # MC_left
                        help='speaker3_video')
    parser.add_argument('--speaker3_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/MC_right.mp4', # MC_right
                        help='speaker3_video')
    
    # parser.add_argument('--transcript_file', type=str, default='/NasData/home/ych/multi-channel-video-compile-2024/transcription.json')
    parser.add_argument('--weights', type=str, default='/NasData/home/ych/multi-channel-video-compile-2024/snapshot_swin_v2_b_2_0.9563032640482664.pth')
    args = parser.parse_args()

    # segments = parse_transcript(args.transcript_file) 

    # sorted_segments_by_duration = sorted(segments.items(), 
    #                                      key=lambda x: x[1]['end'] - x[1]['start'], reverse=True)

    
    tmp_cap = cv2.VideoCapture(args.widechannel_video)
    total_frames = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(tmp_cap.get(cv2.CAP_PROP_FPS))
    tmp_cap.release()

    speaker_names = ['C', 'D', 'MC_left', 'MC_right']
    spkr_video_paths = [args.speaker0_video, args.speaker1_video, args.speaker2_video, args.speaker3_video]

    speakers = [
        Speaker(
            name=name,
            video_path=video_path,
            queue=mp.Queue(),
            result_list=mp.Manager().list([None for _ in range(total_frames)])
        )
        for name, video_path in zip(speaker_names, spkr_video_paths)
    ]

    for speaker in speakers:
        speaker.producer = mp.Process(target=lip_detection_in_video, 
                                    args=(speaker.video_path, speaker.queue, total_frames))
        speaker.producer.start()
        print(f"{speaker.name} producer started")

    for speaker in speakers:
        speaker.consumer = mp.Process(target=infer_lip_state, 
                                    args=(speaker.queue, speaker.result_list, 'swin_v2_b', args.weights))
        speaker.consumer.start()
        print(f"{speaker.name} consumer started")

    for speaker in speakers:
        speaker.producer.join()
        print(f"{speaker.name} producer joined")

    for speaker in speakers:
        speaker.queue.put('LAST')
        print(f"{speaker.name} sentinel value sent")

    for speaker in speakers:
        speaker.consumer.join()
        print(f"{speaker.name} consumer joined")

    result_dict = {}
    for i in range(total_frames):
        frame_result = {}
        for speaker in speakers:
            entry = speaker.result_list[i]
            if isinstance(entry, tuple) and len(entry) == 2:
                state, prob = entry 
            else:
                state, prob = "None", "None"

            frame_result[speaker.name] = {'prob': prob}
        frame_result["max_prob_channel"] = find_max_prob_channel(frame_result)
        result_dict[int(i)] = frame_result

    with open("./multi_channel_lip_infer.json", 'w') as f:
        json.dump(result_dict, f, indent=4)