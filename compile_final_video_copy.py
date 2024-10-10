import json
import cv2
import time
import os
from moviepy.editor import AudioFileClip, VideoFileClip
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
from pprint import pprint
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

    # print(trans_list)

    segments = {}

    for i, item in enumerate(trans_list):
        try:
            speaker = str(item['speaker'])
        except:
            speaker = "UNKNOWN"

        start_time = float(item['start'])
        end_time = float(item['end'])
        start_frame = time_to_frames(start_time)
        end_frame = time_to_frames(end_time)


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

def load_data(multi_channel_file, transcription_file):
    with open(multi_channel_file) as f:
        multi_channel_data = json.load(f)

    with open(transcription_file) as f:
        transcription_data = json.load(f)
    
    return multi_channel_data, transcription_data

def map_speaker_to_max_prob_channel(multi_channel_data, transcription_data, fps=30):
    # speaker 별로 max_prob_channel을 저장할 딕셔너리 생성
    speaker_channel_map = defaultdict(list)
    last_known_channel = None

    # transcription_data를 순회하며 segment에 해당하는 speaker와 max_prob_channel을 매핑
    for segment in transcription_data:
        if isinstance(segment, dict):
            speaker = segment.get("speaker")
            if not speaker and last_known_channel:  # speaker가 없을 경우 이전 채널 사용
                speaker = last_known_channel
            elif speaker:
                last_known_channel = speaker

            start_frame = int(segment.get("start", 0) * fps)  # 초를 프레임으로 변환, 기본값 0
            end_frame = int(segment.get("end", 0) * fps)  # 초를 프레임으로 변환, 기본값 0
            
            # start_frame에서 end_frame까지의 max_prob_channel을 수집
            for frame in range(start_frame, end_frame + 1):
                if str(frame) in multi_channel_data:
                    max_prob_channel = multi_channel_data[str(frame)]
                    speaker_channel_map[speaker].append(max_prob_channel)

    # speaker별 max_prob_channel의 최빈값을 계산하여 매칭
    speaker_max_prob_mapping = {}
    for speaker, channels in speaker_channel_map.items():
        if channels:
            most_common_channel = Counter(channels).most_common(1)[0][0]
            speaker_max_prob_mapping[speaker] = most_common_channel
        else:
            speaker_max_prob_mapping[speaker] = "No channel data"

    return speaker_max_prob_mapping

def map_frames_to_speakers(transcription_data, fps=30):
    # 프레임 번호를 리스트의 인덱스로 간주하고, 해당 프레임의 speaker를 저장하는 리스트 생성
    frame_speaker_list = []
    last_known_speaker = "UNKNOWN_SPEAKER"

    for segment in transcription_data:
        if isinstance(segment, dict):
            speaker = segment.get("speaker", last_known_speaker)
            if speaker != "UNKNOWN_SPEAKER":
                last_known_speaker = speaker

            start_frame = int(segment.get("start", 0) * fps)  # 초를 프레임으로 변환, 기본값 0
            end_frame = int(segment.get("end", 0) * fps)  # 초를 프레임으로 변환, 기본값 0

            # 프레임 번호에 해당하는 speaker 저장
            while len(frame_speaker_list) <= end_frame:
                frame_speaker_list.append("UNKNOWN_SPEAKER")

            for frame in range(start_frame, end_frame + 1):
                frame_speaker_list[frame] = speaker

    return frame_speaker_list

def reverse_dict(input_dict):
    """
    Given a dictionary, return a new dictionary with keys and values swapped.
    
    Args:
    input_dict (dict): The dictionary to reverse.
    
    Returns:
    dict: A new dictionary with keys and values swapped.
    """
    return {value: key for key, value in input_dict.items()}

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
    parser.add_argument('--speaker_videos', type=str, nargs='+',
                    default=[
                        '/NasData/home/ych/Multicam_materials/thelive/C.mp4', 
                        '/NasData/home/ych/Multicam_materials/thelive/D.mp4', 
                        '/NasData/home/ych/Multicam_materials/thelive/MC_left.mp4', 
                        '/NasData/home/ych/Multicam_materials/thelive/MC_right.mp4'
                    ],
                    help='List of speaker videos')
    
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    
    parser.add_argument('--transcript_file', type=str, 
                        default='/NasData/home/ych/multi-channel-video-compile-2024/transcriptions_.json')
    parser.add_argument('--channel_inference_file', type=str, 
                        default='/NasData/home/ych/multi-channel-video-compile-2024/multi_channel_lip_infer_exp3.json')
    
    parser.add_argument('--final_video_path', type=str, 
                        default='/NasData/home/ych/multi-channel-video-compile-2024/compiled_sample/sample_thelive.mp4',
                        help='final video path') 
    args = parser.parse_args()

  

    # segments = parse_transcript(args.transcript_file)
    # channel_infer = parse_channel_inference(args.channel_inference_file)
    # print(segments)
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
   
    multi_channel_data, transcription_data = load_data(args.channel_inference_file, args.transcript_file)
    speaker_max_prob_mapping = map_speaker_to_max_prob_channel(multi_channel_data, transcription_data)
    print(speaker_max_prob_mapping)
    speaker_max_prob_mapping = reverse_dict(speaker_max_prob_mapping)
    print(speaker_max_prob_mapping)
                

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
    # exit()
    
    widechannel_video = cv2.VideoCapture(args.widechannel_video)
    spkr_video_paths = args.speaker_videos
    audio = AudioFileClip(args.widechannel_video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video = cv2.VideoWriter(args.final_video_path, fourcc, 30, (1920, 1080))

    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame else int(widechannel_video.get(cv2.CAP_PROP_FRAME_COUNT))


    for i, spkr_video_path in enumerate(spkr_video_paths):
        locals()[f'speaker{i+1}_video'] = cv2.VideoCapture(spkr_video_path)
        

    selected_channels = map_frames_to_speakers(transcription_data)
    selected_channels = adjust_abnormal_channels(selected_channels)

    @dataclass
    class origin_video:
        video_path: str
        video: cv2.VideoCapture
        speaker: str

    origin_videos = [
        origin_video(video_path=p, 
                     video=cv2.VideoCapture(p),
                     speaker=speaker_max_prob_mapping[p.split('/')[-1]]
                     ) for p in spkr_video_paths
    ]
    pprint(origin_videos)      
    exit()

    while all(v.video.isOpened() for v in origin_videos) and widechannel_video.isOpened():
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
            for v in origin_videos:
                v.video.release()
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
        