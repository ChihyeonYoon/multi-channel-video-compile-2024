import json
import cv2
import time
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pprint import pprint
import moviepy.editor
from pydub import AudioSegment

def frame_number_to_hhmmss(frame_number, frames_per_second=30):
    total_seconds = frame_number / frames_per_second
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def seconds_to_hhmmss(seconds):
    """
    초를 HH:MM:SS 형식으로 변환하는 함수.
    
    :param seconds: 초 단위 시간
    :return: HH:MM:SS 형식의 문자열
    """
    hours = seconds // 3600  # 시 계산
    minutes = (seconds % 3600) // 60  # 분 계산
    seconds = seconds % 60  # 초 계산
    
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

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
    

def smooth_short_segments(speaker_data, threshold=30):

    # 배열로 변환
    speaker_data = np.array(speaker_data)
    n = len(speaker_data)

    # 현재 구간의 시작 인덱스를 저장하는 변수
    segment_start = 0

    # 전체 데이터를 순회하며 구간 나누기
    for i in range(1, n + 1):
        # 끝에 도달했거나 화자가 바뀌는 경우 현재 구간을 평가합니다.
        if i == n or speaker_data[i] != speaker_data[segment_start]:
            segment_length = i - segment_start

            # 현재 구간의 길이가 threshold보다 작다면 앞뒤 화자 값으로 채우기
            if segment_length <= threshold:
                left_speaker = speaker_data[segment_start - 1] if segment_start > 0 else None
                right_speaker = speaker_data[i] if i < n else None

                # 가능한 경우 앞쪽 화자 값으로 채우기, 아니면 뒤쪽 화자 값 사용
                if left_speaker is not None and left_speaker == right_speaker:
                    speaker_data[segment_start:i] = left_speaker
                elif left_speaker is not None:
                    speaker_data[segment_start:i] = left_speaker
                elif right_speaker is not None:
                    speaker_data[segment_start:i] = right_speaker

            # 다음 구간의 시작 인덱스를 업데이트합니다.
            segment_start = i

    return speaker_data.tolist()

def load_data(multi_channel_file, transcription_file):
    with open(multi_channel_file) as f:
        multi_channel_data = json.load(f)

    with open(transcription_file) as f:
        transcription_data = json.load(f)
    
    return multi_channel_data, transcription_data

def plot_segments(speaker_data, fps=30, filename='speaker_segments.png'):
    # 각 구간의 길이와 화자 정보 추출
    segments = []
    start = 0
    current_speaker = speaker_data[0]

    for i in range(1, len(speaker_data)):
        if speaker_data[i] != current_speaker:
            segments.append((current_speaker, start, i - start))
            start = i
            current_speaker = speaker_data[i]
    segments.append((current_speaker, start, len(speaker_data) - start))

    # 색상 코드 정의
    color_palette = {
        "SPEAKER_00": "#264653",
        "SPEAKER_01": "#2A9D8F",
        "SPEAKER_02": "#E9C46A",
        "SPEAKER_03": "#F4A261",
        "UNKNOWN_SPEAKER": "#E76F51"
    }

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    y_labels = []
    y_positions = {}

    for idx, (speaker, start, length) in enumerate(segments):
        if speaker not in y_positions:
            y_positions[speaker] = len(y_labels)
            y_labels.append(speaker)
        color = color_palette.get(speaker, "#000000")  # 지정되지 않은 화자의 경우 기본 색상 검정색
        ax.barh(y_positions[speaker], length / fps, left=start / fps, color=color)

    sorted_y_labels = y_labels
    ax.set_yticks(range(len(sorted_y_labels)))
    ax.set_yticklabels(sorted_y_labels)  # y축 성분 유지
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Speaker Segments Over Time')
    plt.tight_layout()
    plt.savefig(filename)

# 두번째 함수
def plot_segments2(speaker_data, fps=30, filename='speaker_segments2.png'):
    # 각 구간의 길이와 화자 정보 추출
    segments = []
    start = 0
    current_speaker = speaker_data[0]

    for i in range(1, len(speaker_data)):
        if speaker_data[i] != current_speaker:
            segments.append((current_speaker, start, i - start))
            start = i
            current_speaker = speaker_data[i]
    segments.append((current_speaker, start, len(speaker_data) - start))

    # 색상 코드 정의
    color_palette = {
        "SPEAKER_00": "#264653",
        "SPEAKER_01": "#2A9D8F",
        "SPEAKER_02": "#E9C46A",
        "SPEAKER_03": "#F4A261",
        "UNKNOWN_SPEAKER": "#E76F51"
    }

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 3))  # 세로 길이 축소

    for idx, (speaker, start, length) in enumerate(segments):
        color = color_palette.get(speaker, "#000000")  # 지정되지 않은 화자의 경우 기본 색상 검정색
        ax.barh(0, length / fps, left=start / fps, color=color, label=speaker if speaker not in ax.get_legend_handles_labels()[1] else "")

    ax.set_yticks([0])
    ax.set_yticklabels(['Speakers'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Speaker Segments Over Time')
    ax.set_ylim(-1, 1)  # 상하 여백 추가
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Speakers', loc='upper right', ncol=1)  # 범례를 화자별 색상에 맞춰 일관되게 유지
    plt.tight_layout()
    plt.savefig(filename)

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

def add_trimmed_audio_to_video(start_time, end_time, final_video_path, audio_path, output_path):
    # Load original audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Convert start and end times from seconds to milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    
    # Trim audio
    trimmed_audio = audio[start_ms:end_ms]
    
    # Save the trimmed audio to a temporary file
    temp_audio_path = "temp_trimmed_audio.wav"
    trimmed_audio.export(temp_audio_path, format="wav")
    
    # Load the video
    video = moviepy.editor.VideoFileClip(final_video_path)
    
    # Load the trimmed audio
    audio_clip = moviepy.editor.AudioFileClip(temp_audio_path)
    
    # Set the audio to the video
    video_with_audio = video.set_audio(audio_clip)
    
    # Write the final output video
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Remove the temporary audio file
    os.remove(temp_audio_path)


if __name__ == '__main__':
    # 38101 frames
    parser = argparse.ArgumentParser()
    parser.add_argument('--wide_ch_video', type=str, 
                        default='./materials/thelive/W.mp4',
                        help='widechannel_video')
    parser.add_argument('--speaker_ch_videos', type=str, nargs='+',
                    default=[
                        './materials/thelive/C.mp4', 
                        './materials/thelive/D.mp4', 
                        './materials/thelive/MC_left.mp4', 
                        './materials/thelive/MC_right.mp4'
                    ],
                    help='List of speaker videos')
    parser.add_argument('--audio_path', type=str,
                        default='./materials/thelive/audio.wav',)
    
    parser.add_argument('--start_time', type=int, default=0,
                        help='Start time of the segment to process (in seconds)')
    parser.add_argument('--end_time', type=int, default=0,
                        help='End time of the segment to process (in seconds)')
    
    parser.add_argument('--transcript_path', type=str, 
                        default='/compiled_sample/transcriptions.json')
    parser.add_argument('--channel_inference_file', type=str, 
                        default='/compiled_sample/multi_channel_lip_infer_exp.json')
    
    parser.add_argument('--save_path', type=str,
                        default='./compiled_sample',
                        help='Path to save the final video')
    args = parser.parse_args()

    run_start = time.time()
   
    multi_channel_data, transcription_data = load_data(args.channel_inference_file, args.transcript_path)
    speaker_max_prob_mapping = map_speaker_to_max_prob_channel(multi_channel_data, transcription_data)
    speaker_max_prob_mapping = reverse_dict(speaker_max_prob_mapping)
    print(speaker_max_prob_mapping)

    # Optional in this case. You should disable this block if you are on normal case.
    for k in speaker_max_prob_mapping.keys():
        if k.endswith("_.mp4"):
            speaker_max_prob_mapping[k.replace("_.mp4", ".mp4")]=speaker_max_prob_mapping.pop(k)
    print("Optional changes","\n",speaker_max_prob_mapping)
                
    widechannel_video = cv2.VideoCapture(args.wide_ch_video)
    spkr_video_paths = args.speaker_ch_videos

    # FPS 정보 가져오기
    fps = widechannel_video.get(cv2.CAP_PROP_FPS)

    # start_time과 end_time을 초 단위에서 프레임 단위로 변환
    start_frame = time_to_frames(args.start_time, fps)
    end_frame = time_to_frames(args.end_time, fps) if args.end_time else int(widechannel_video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i, spkr_video_path in enumerate(spkr_video_paths):
        locals()[f'speaker{i+1}_video'] = cv2.VideoCapture(spkr_video_path)
        
    infer_selected_channels = json.load(open(args.channel_inference_file))
    infer_selected_channels = [infer_selected_channels[str(i)] for i in range(len(infer_selected_channels))]
    selected_channels = [speaker_max_prob_mapping.get(channel, 'UNKNOWN_SPEAKER') for channel in infer_selected_channels]

    trans_selected_channels = map_frames_to_speakers(transcription_data)
    trans_selected_channels = smooth_short_segments(trans_selected_channels)

    selected_channels[:len(trans_selected_channels)-1] = trans_selected_channels

    # print(selected_channels)
    plot_segments(selected_channels, filename=args.save_path+'/speaker_segments.png')
    plot_segments2(selected_channels, filename=args.save_path+'/speaker_segments2.png')
   
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
    origin_videos.append(origin_video(video_path=args.wide_ch_video,
                                      video=widechannel_video,
                                      speaker='widechannel')
                        )
    origin_videos = sorted(origin_videos, key=lambda x: x.speaker)
    pprint(origin_videos)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video_path = args.save_path + "/sample.mp4"
    final_video = cv2.VideoWriter(final_video_path, fourcc, fps, (1920, 1080))
    
    # 비디오의 시작을 start_frame으로 설정
    for v in origin_videos:
        v.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while True:
        all_videos_have_frames = False
        frames = {}

        # 모든 비디오에 대해 프레임을 읽기
        for v in origin_videos:
            ret, frame = v.video.read()
            if ret:
                all_videos_have_frames = True
                frames[v.speaker] = frame
            else:
                print(f"No more frames in video {v.speaker}")
                frames[v.speaker] = None

        # 만약 모든 비디오에서 더 이상 읽을 수 있는 프레임이 없다면 종료
        if not all_videos_have_frames:
            break

        # 현재 프레임의 인덱스
        current_frame = int(origin_videos[0].video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        # 설정된 end_frame에 도달하면 종료
        if current_frame >= end_frame:
            break

        # 현재 프레임의 speaker와 일치하는 비디오의 프레임을 final_video에 작성
        if current_frame >= len(selected_channels):
            break

        current_speaker = selected_channels[current_frame]

        # 해당하는 speaker의 비디오 프레임이 있으면 final_video에 작성
        if current_speaker in frames and frames[current_speaker] is not None:
            final_video.write(frames[current_speaker])
        elif 'widechannel' in frames and frames['widechannel'] is not None:
            # UNKNOWN_SPEAKER에 해당하는 경우 widechannel 프레임 사용
            final_video.write(frames['widechannel'])
        
        if current_frame % 100 == 0:
            print(f"Processing frame {current_frame}/{end_frame}")

    # 모든 창 닫기 및 비디오 객체 해제
    cv2.destroyAllWindows()
    final_video.release()
    for v in origin_videos:
        v.video.release()

    add_trimmed_audio_to_video(args.start_time, args.end_time, 
                               final_video_path, args.audio_path, 
                               final_video_path.replace('.mp4', '_with_audio.mp4'))

