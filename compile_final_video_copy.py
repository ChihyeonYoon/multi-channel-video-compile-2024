import json
from tracemalloc import start
import cv2
import time
import os
from more_itertools import first
from moviepy.editor import AudioFileClip, VideoFileClip
from collections import Counter
import argparse
import re

import scipy as sp

# from matching_the_speaker import SpeakerMatcher

def frame_number_to_hhmmss(frame_number, frames_per_second=30):
    total_seconds = frame_number / frames_per_second
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def time_to_frames(time_in_seconds, frames_per_second=30):
    return int(time_in_seconds * frames_per_second)

def time_to_seconds(time_str):
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

def get_frame_numbers(start_time, end_time, frame_rate):
    # Calculate the start and end frame numbers
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)

    # Generate a list of all frame numbers in the range
    frame_numbers = list(range(start_frame, end_frame))

    return frame_numbers
"""
def parse_transcript(file_path):
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()

    with open(file_path, 'r', encoding='utf-8') as file:
        trans_list = json.load(file)
    
    
    speaker_segments = {}

    for item in trans_list:
        start_time = float(item['start'])
        end_time = float(item['end'])
        speaker = str(item['speaker'])
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        
        frame_numbers = get_frame_numbers(float(start_time), float(end_time), 30)
        speaker_segments[speaker] += frame_numbers
    
    return speaker_segments
"""

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
        
    """
    segments = {idx: {'start': start_time, 
            'end': end_time, 
            'start_frame': start_frame, 
            'end_frame': end_frame, 
            'speaker': speaker}
            }
    """
    
    return segments

def parse_channel_inference(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        channel_inference = json.load(file)
    """
    channel_infer = {frame_number-1 (str): channel:{'prob': [slince, utterance]},
                    max_prob_channel: str
                    }
    """
    return channel_inference 
    

def adjust_abnormal_channels(channels, abnormal_value="widechannel", fps=30):
    adjusted_channels = channels.copy()
    length = len(channels)
    i = 0

    while i < length:
        if channels[i] == abnormal_value:
            start = i
            while i < length and channels[i] == abnormal_value:
                i += 1
            end = i

            if end - start < fps:
                previous_value = channels[start - 1] if start > 0 else None
                next_value = channels[end] if end < length else None
                replacement_value = previous_value if previous_value is not None else next_value

                for j in range(start, end):
                    adjusted_channels[j] = replacement_value
        else:
            i += 1

    return adjusted_channels

def match_speaker_to_unique_channel(channel_infer_file, diarization_file, fps=30):
    # JSON 파일 로드
    with open(channel_infer_file, 'r') as f:
        channel_infer = json.load(f)

    with open(diarization_file, 'r') as f:
        diarization = json.load(f)

    # 프레임 변환 함수
    def time_to_frame(time_in_seconds, fps):
        return int(time_in_seconds * fps)

    # 각 화자별로 채널 누적 확률을 저장할 딕셔너리
    speaker_channel_probabilities = {}

    # 다이어리제이션의 각 구간을 화자별로 채널 매칭
    for segment in diarization:
        speaker = segment['speaker']
        start_time = segment['start']
        end_time = segment['end']

        # 시간 구간을 프레임으로 변환
        start_frame = time_to_frame(start_time, fps)
        end_frame = time_to_frame(end_time, fps)

        # 화자가 새로 등장하면 초기화
        if speaker not in speaker_channel_probabilities:
            speaker_channel_probabilities[speaker] = {}

        # 해당 시간 구간의 프레임에서 채널 확률 확인
        for frame in range(start_frame, end_frame + 1):
            frame_str = str(frame)  # JSON의 키가 문자열로 되어 있음
            if frame_str in channel_infer:
                for channel, prob in channel_infer[frame_str].items():
                    if channel == 'max_prob_channel':
                        continue  # max_prob_channel은 제외
                    if prob is not None:
                        if channel not in speaker_channel_probabilities[speaker]:
                            speaker_channel_probabilities[speaker][channel] = 0
                        speaker_channel_probabilities[speaker][channel] += prob[-1]  # 첫 번째 확률 값만 사용

    # 채널 할당을 위한 딕셔너리
    speaker_best_channels = {}
    used_channels = set()  # 이미 사용된 채널을 저장하는 집합

    # 각 화자별로 가장 확률이 높은 채널 선택 (중복 방지)
    for speaker, channels in speaker_channel_probabilities.items():
        sorted_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)  # 확률이 높은 순으로 정렬
        for channel, _ in sorted_channels:
            if channel not in used_channels:
                speaker_best_channels[speaker] = channel
                used_channels.add(channel)
                break  # 첫 번째로 가능한 채널을 할당하고 종료

    # 결과를 딕셔너리로 반환
    return speaker_best_channels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 38101 frames
    # parser.add_argument('--widechannel_video', type=str, 
    #                     default='/NasData/home/ych/2024_Multicam/materials/thelive/W.mp4',
    #                     help='widechannel_video')
    # parser.add_argument('--speaker1_video', type=str, 
    #                     default='/NasData/home/ych/2024_Multicam/materials/thelive/C.mp4', # C
    #                     help='speaker1_video')
    # parser.add_argument('--speaker2_video', type=str, 
    #                     default='/NasData/home/ych/2024_Multicam/materials/thelive/D.mp4', # D
    #                     help='speaker2_video')
    # parser.add_argument('--speaker3_video', type=str, 
    #                     default='/NasData/home/ych/2024_Multicam/materials/thelive/MC.mp4', # MC
    #                     help='speaker3_video')

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
                        default='/NasData/home/ych/Multicam_materials/thelive/MC_left.mp4', # MC
                        help='speaker3_video')
    parser.add_argument('--speaker3_video', type=str, 
                        default='/NasData/home/ych/Multicam_materials/thelive/MC_left.mp4', # MC
                        help='speaker3_video')
    
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    
    parser.add_argument('--transcript_file', type=str, default='/NasData/home/ych/multi-channel-video-compile-2024/transcription.json')
    parser.add_argument('--channel_inference_file', type=str, default='/NasData/home/ych/multi-channel-video-compile-2024/multi_channel_lip_infer.json')
    parser.add_argument('--final_video_path', type=str, default='/NasData/home/ych/multi-channel-video-compile-2024/compiled_sample/sample_thelive.mp4',
                        help='final video path') 
    args = parser.parse_args()

    file_path = args.transcript_file

    segments = parse_transcript(file_path)
    channel_infer = parse_channel_inference(args.channel_inference_file)
    # print(channel_infer)
    # exit()
    """
    segments = {idx: {'start': start_time, 
            'end': end_time, 
            'start_frame': start_frame, 
            'end_frame': end_frame, 
            'speaker': speaker}
            }

    channel_infer = {frame_number-1 (str): channel:{'prob': [slince, utterance]},
                    max_prob_channel: str
                    }
    """
    best_channels = match_speaker_to_unique_channel(channel_infer_file="/NasData/home/ych/multi-channel-video-compile-2024/multi_channel_lip_infer_exp1.json", 
                                 diarization_file= "/NasData/home/ych/multi-channel-video-compile-2024/transcription.json", 
                                 fps=30)
    print(best_channels)
    best_channels2 = match_speaker_to_unique_channel(channel_infer_file="/NasData/home/ych/multi-channel-video-compile-2024/multi_channel_lip_infer_exp2.json",
                                    diarization_file= "/NasData/home/ych/multi-channel-video-compile-2024/transcription.json",
                                    fps=30)
    print(best_channels2)

                

    # spker0_first_segment = next((segments[i] for i in segments.keys() if segments[i]['speaker'] == 'SPEAKER_00'), None)
    # spker1_first_segment = next((segments[i] for i in segments.keys() if segments[i]['speaker'] == 'SPEAKER_01'), None)
    # spker2_first_segment = next((segments[i] for i in segments.keys() if segments[i]['speaker'] == 'SPEAKER_02'), None)
    # spker3_first_segment = next((segments[i] for i in segments.keys() if segments[i]['speaker'] == 'SPEAKER_03'), None)

    # spker0_last_segment = next((segments[i] for i in reversed(segments.keys()) if segments[i]['speaker'] == 'SPEAKER_00'), None) 
    # spker1_last_segment = next((segments[i] for i in reversed(segments.keys()) if segments[i]['speaker'] == 'SPEAKER_01'), None)
    # spker2_last_segment = next((segments[i] for i in reversed(segments.keys()) if segments[i]['speaker'] == 'SPEAKER_02'), None)
    # spker3_last_segment = next((segments[i] for i in reversed(segments.keys()) if segments[i]['speaker'] == 'SPEAKER_03'), None)

    # print(f"Speaker 0: {spker0_first_segment['start']} {spker0_first_segment['end']} | {spker0_first_segment['start_frame']} - {spker0_first_segment['end_frame']}") if spker0_first_segment else None
    # print(f"Speaker 0: {spker0_last_segment['start']} {spker0_last_segment['end']} | {spker0_last_segment['start_frame']} - {spker0_last_segment['end_frame']}") if spker0_last_segment else None

    # print(f"Speaker 1: {spker1_first_segment['start']} {spker1_first_segment['end']} | {spker1_first_segment['start_frame']} - {spker1_first_segment['end_frame']}") if spker1_first_segment else None
    # print(f"Speaker 1: {spker1_last_segment['start']} {spker1_last_segment['end']} | {spker1_last_segment['start_frame']} - {spker1_last_segment['end_frame']}") if spker1_last_segment else None
    
    # print(f"Speaker 2: {spker2_first_segment['start']} {spker2_first_segment['end']} | {spker2_first_segment['start_frame']} - {spker2_first_segment['end_frame']}") if spker2_first_segment else None
    # print(f"Speaker 2: {spker2_last_segment['start']} {spker2_last_segment['end']} | {spker2_last_segment['start_frame']} - {spker2_last_segment['end_frame']}") if spker2_last_segment else None
    
    # print(f"Speaker 3: {spker3_first_segment['start']} {spker3_first_segment['end']} | {spker3_first_segment['start_frame']} - {spker3_first_segment['end_frame']}") if spker3_first_segment else None
    # print(f"Speaker 3: {spker3_last_segment['start']} {spker3_last_segment['end']} | {spker3_last_segment['start_frame']} - {spker3_last_segment['end_frame']}") if spker3_last_segment else None

    # first_segments = [
    #     list(map(time_to_frames, [spker0_first_segment['start'], spker0_first_segment['end']])), 
    #     list(map(time_to_frames, [spker1_first_segment['start'], spker1_first_segment['end']])), 
    #     list(map(time_to_frames, [spker2_first_segment['start'], spker2_first_segment['end']])),
    #     list(map(time_to_frames, [spker3_first_segment['start'], spker3_first_segment['end']]))
    #     ]

    # first_segments = [
    #     [spker0_first_segment['start_frame'], spker0_first_segment['end_frame']],
    #     [spker1_first_segment['start_frame'], spker1_first_segment['end_frame']],
    #     [spker2_first_segment['start_frame'], spker2_first_segment['end_frame']],
    #     [spker3_first_segment['start_frame'], spker3_first_segment['end_frame']]
    # ]

    # print(first_segments)
    # video_list = ['/NasData/home/ych/2024_Multicam/materials/thelive/MC_left.mp4',
    #               '/NasData/home/ych/2024_Multicam/materials/thelive/C.mp4',
    #               '/NasData/home/ych/2024_Multicam/materials/thelive/D.mp4',
    #               '/NasData/home/ych/2024_Multicam/materials/thelive/MC_right.mp4']
    
    # SpeakerMatcher = SpeakerMatcher(video_list=video_list, segments=first_segments)
    # SpeakerMatcher.match_speaker()
    exit()
    
    widechannel_video = cv2.VideoCapture(args.widechannel_video)
    speaker1_video = cv2.VideoCapture(args.speaker2_video) # C
    speaker2_video = cv2.VideoCapture(args.speaker1_video) # D
    speaker3_video = cv2.VideoCapture(args.speaker3_video) # MC
    audio = AudioFileClip(args.widechannel_video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video = cv2.VideoWriter(args.final_video_path, fourcc, 30, (1920, 1080))

    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame else min(int(widechannel_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                                                              int(speaker1_video.get(cv2.CAP_PROP_FRAME_COUNT)),  
                                                              int(speaker2_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                                                              int(speaker3_video.get(cv2.CAP_PROP_FRAME_COUNT)))


    selected_channels = [0 for _ in range(end_frame)]
    for i, v in enumerate(selected_channels):
        if i in segments['SPEAKER_00'] or i in segments['SPEAKER_03']:
            selected_channels[i] = '3'
        elif i in segments['SPEAKER_02']:
            selected_channels[i] = '1'
        elif i in segments['SPEAKER_01']:
            selected_channels[i] = '2'
        else:
            selected_channels[i] = 'widechannel'
    
    selected_channels = adjust_abnormal_channels(selected_channels)

    

    while(widechannel_video.isOpened() and speaker1_video.isOpened() and speaker2_video.isOpened() and speaker3_video.isOpened()):
        retw, frame_w = widechannel_video.read()
        ret1, frame_1 = speaker1_video.read() # C
        ret2, frame_2 = speaker2_video.read() # D
        ret3, frame_3 = speaker3_video.read() # MC
        current_frame = int(widechannel_video.get(cv2.CAP_PROP_POS_FRAMES))

        try:
            if start_frame <= current_frame <= end_frame:
                print(f"Frame: {current_frame}/{end_frame}")

                if selected_channels[current_frame] == '1':
                    final_video.write(frame_1)
                elif selected_channels[current_frame] == '2':
                    final_video.write(frame_2)
                elif selected_channels[current_frame] == '3':
                    final_video.write(frame_3)
                else:
                    final_video.write(frame_w)
        except Exception as e:
            # print(e)
            break

        if current_frame >= end_frame:
            widechannel_video.release()
            speaker1_video.release()
            speaker2_video.release()
            speaker3_video.release()
            break
    
    final_video.release()
    print('Vdieo compilation is done')

    audio = audio.subclip(start_frame/30, end_frame/30)
    final_video_with_audio = VideoFileClip(args.final_video_path)
    final_video_with_audio = final_video_with_audio.set_audio(audio)
    final_video_with_audio.write_videofile(args.final_video_path.replace('.mp4', '_with_audio_adj.mp4'), codec='libx264', audio_codec='aac')
    os.remove(args.final_video_path)

    print(f'output video saved at {args.final_video_path.replace(".mp4", "_with_audio_adj.mp4")}')

    tmpdict = {i+1: v for i, v in enumerate(selected_channels)}
    with open('selected_frames_adj.json', 'w') as f:
        json.dump(tmpdict, f)
        