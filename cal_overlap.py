import json
import matplotlib.pyplot as plt

# 파일 불러오기
with open('/NasData/home/ych/multi-channel-video-compile-2024/materials/thelive/label_thelive.json', 'r') as f:
    label_data = json.load(f)

with open('/NasData/home/ych/multi-channel-video-compile-2024/outputs/selected_channels.json', 'r') as f:
    selected_channels = json.load(f)

# 일치율 계산 함수
def calculate_overlap(label_data, selected_channels):
    results = []
    total_matching_frames = 0
    total_frames = 0

    for segment in label_data:
        frames = segment['frames']
        segment_start_time = segment['start_time']
        segment_end_time = segment['end_time']
        
        # Segment 범위 내 selected_channels의 frame 추출
        selected_segment_frames = {k: v for k, v in selected_channels.items() if k in frames}
        
        # 교집합 구하기
        intersection = {
            frame: speaker for frame, speaker in frames.items()
            if frame in selected_segment_frames and selected_segment_frames[frame] == speaker
        }

        # Segment 별 일치율 계산
        segment_total_frames = len(frames)
        segment_match_frames = len(intersection)
        segment_overlap_ratio = segment_match_frames / segment_total_frames if segment_total_frames > 0 else 0

        # 결과 저장
        results.append({
            "start_time": segment_start_time,
            "end_time": segment_end_time,
            "overlap_ratio": segment_overlap_ratio,
            "matching_frames": segment_match_frames,
            "total_frames": segment_total_frames
        })

        # 전체 계산에 추가
        total_matching_frames += segment_match_frames
        total_frames += segment_total_frames

    # 전체 segment에 걸친 일치율 계산
    overall_overlap_ratio = total_matching_frames / total_frames if total_frames > 0 else 0

    return results, overall_overlap_ratio

overlap_results, overall_overlap_ratio = calculate_overlap(label_data, selected_channels)

overlap_results = sorted(overlap_results, key=lambda x: x['start_time'])

for result in overlap_results:
    print(f"Segment ({result['start_time']} - {result['end_time']}) - 일치율: {result['overlap_ratio']:.2f}")

print(f"전체 segment에 걸친 일치율: {overall_overlap_ratio:.2f}\n")

time_points = [(segment['start_time'] + segment['end_time']) / 2 for segment in overlap_results]
overlap_ratios = [segment['overlap_ratio'] for segment in overlap_results]


plt.figure(figsize=(15, 4))
plt.plot(time_points, overlap_ratios, marker='o', linestyle='-', label='Segment Overlap Ratio')
plt.axhline(y=overall_overlap_ratio, color='r', linestyle='--', label=f'Overall Overlap Ratio: {overall_overlap_ratio:.2f}')

# 그래프 설정
plt.xlabel('Time (seconds)')
plt.ylabel('Overlap Ratio')
plt.title('Overlap Ratio over Time')
plt.legend()
plt.grid(True)
plt.savefig('overlap_ratio.png')


from collections import defaultdict
def calculate_speaker_overlap(label_data, selected_channels):
    speaker_results = defaultdict(lambda: {"matching_frames": 0, "total_frames": 0})
    speaker_overlap_ratios = {}  # 최종 일치율을 저장할 딕셔너리
    
    for segment in label_data:
        frames = segment['frames']
        
        # segment 내 selected_channels의 frame 추출
        selected_segment_frames = {k: v for k, v in selected_channels.items() if k in frames}
        
        # 각 speaker 별로 일치하는 frame 수와 전체 frame 수를 집계
        for frame, speaker in frames.items():
            if frame in selected_segment_frames:
                if selected_segment_frames[frame] == speaker:
                    speaker_results[speaker]["matching_frames"] += 1
                speaker_results[speaker]["total_frames"] += 1

    # speaker 별 일치율 계산 및 결과 저장
    for speaker, counts in speaker_results.items():
        matching = counts["matching_frames"]
        total = counts["total_frames"]
        speaker_overlap_ratios[speaker] = matching / total if total > 0 else 0

    return speaker_overlap_ratios, speaker_results

# 결과 계산 및 출력
speaker_overlap_ratios, speaker_results = calculate_speaker_overlap(label_data, selected_channels)
for speaker, ratio in speaker_overlap_ratios.items():
    print(f"Speaker: {speaker}, 일치율: {ratio:.2f}, 일치하는 frame 수: {speaker_results[speaker]['matching_frames']}, 전체 frame 수: {speaker_results[speaker]['total_frames']}")

def calculate_overlap_by_length(label_data, selected_channels):
    results = []

    for segment in label_data:
        start_time = segment['start_time']
        end_time = segment['end_time']
        segment_frames = segment['frames']
        
        # segment 길이 계산
        segment_length = end_time - start_time

        # segment 내 selected_channels의 frame 추출
        matching_frames = 0
        total_frames = len(segment_frames)

        # 일치하는 frame 수 계산
        for frame, speaker in segment_frames.items():
            if frame in selected_channels and selected_channels[frame] == speaker:
                matching_frames += 1

        # 일치율 계산
        overlap_ratio = matching_frames / total_frames if total_frames > 0 else 0

        # 결과 저장
        results.append({
            "segment_length": segment_length,
            "overlap_ratio": overlap_ratio,
            "matching_frames": matching_frames,
            "total_frames": total_frames
        })

    return results

# 결과 계산 및 출력
overlap_by_length_results = calculate_overlap_by_length(label_data, selected_channels)
for result in overlap_by_length_results:
    print(f"Segment Length: {result['segment_length']}s, 일치율: {result['overlap_ratio']:.2f}, "
          f"일치하는 frame 수: {result['matching_frames']}, 전체 frame 수: {result['total_frames']}")

segment_lengths = [result['segment_length'] for result in overlap_by_length_results]
overlap_ratios = [result['overlap_ratio'] for result in overlap_by_length_results]

plt.figure(figsize=(15, 4))
plt.scatter(segment_lengths, overlap_ratios)

# 그래프 설정
plt.xlabel('Segment Length (seconds)')
plt.ylabel('Overlap Ratio')
plt.title('Trend of Overlap Ratio by Segment Length')
plt.grid(True)
plt.savefig('overlap_ratio_by_length.png')

label_data_no_segment = {}
for segment in label_data:
    for frame, speaker in segment['frames'].items():
        label_data_no_segment[frame] = speaker

# selected_channels와 label_data_no_segment의 교집합 계산
def calculate_matching_rate(dict1, dict2, intervals, frame_rate):
    """
    두 개의 딕셔너리에 대해 지정된 구간별 일치율을 계산하는 함수입니다.

    Parameters:
    dict1 (dict): 첫 번째 딕셔너리로, 키는 프레임 번호(문자열)이고 값은 비교할 값입니다.
    dict2 (dict): 두 번째 딕셔너리로, 키는 프레임 번호(문자열)이고 값은 비교할 값입니다.
    intervals (list of tuples): 구간의 리스트로, 각 구간은 (start_time, end_time) 형태의 튜플입니다.
    frame_rate (float): 초당 프레임 수 (FPS)로, 프레임 번호를 시간으로 변환하는 데 사용됩니다.

    Returns:
    dict: 각 구간별 일치율을 담은 딕셔너리로, 키는 구간의 인덱스 또는 'total'이고 값은 일치율입니다.
    """
    matching_rates = {}

    # 프레임 번호를 정수로 변환하고 집합으로 저장
    frames1 = set(int(k) for k in dict1.keys())
    frames2 = set(int(k) for k in dict2.keys())

    # 두 딕셔너리에서 공통으로 존재하는 프레임 번호만 선택
    common_frames = frames1.intersection(frames2)

    if not common_frames:
        matching_rates['total'] = None
        print("공통된 프레임이 없습니다.")
        return matching_rates

    # 전체 구간에 대한 일치율 계산
    total_frames = len(common_frames)
    matching_count = sum(
        1 for frame in common_frames if dict1[str(frame)] == dict2[str(frame)]
    )
    matching_rates['total'] = matching_count / total_frames

    # 공통 프레임의 최소 및 최대 프레임 번호 계산
    min_frame = min(common_frames)
    max_frame = max(common_frames)

    # 지정된 구간별로 일치율 계산
    for idx, (start_time, end_time) in enumerate(intervals):
        # 시간을 프레임 번호로 변환
        start_frame = int(start_time * frame_rate)
        if end_time == 'end':
            end_frame = max_frame + 1  # 마지막 프레임까지 포함하도록 +1
        else:
            end_frame = int(end_time * frame_rate)

        # 해당 구간에 속하는 프레임 번호 선택
        interval_frames = [
            frame for frame in common_frames if start_frame <= frame < end_frame
        ]

        if not interval_frames:
            print(f"구간 {idx} ({start_time}s ~ {end_time}s)에 해당하는 프레임이 없습니다.")
            matching_rates[idx] = None
            continue

        interval_total_frames = len(interval_frames)
        interval_matching_count = sum(
            1 for frame in interval_frames if dict1[str(frame)] == dict2[str(frame)]
        )
        matching_rate = interval_matching_count / interval_total_frames
        matching_rates[idx] = matching_rate

    # 마지막 지정된 구간 이후의 구간 처리
    last_end_time = intervals[-1][1]
    if last_end_time != 'end':
        if isinstance(last_end_time, (int, float)):
            last_end_frame = int(last_end_time * frame_rate)
            if last_end_frame < max_frame:
                # 마지막 지정된 구간 이후부터 마지막 프레임까지의 구간이 존재함
                start_time = last_end_time
                end_time = 'end'
                start_frame = last_end_frame
                end_frame = max_frame + 1  # 마지막 프레임까지 포함

                # 해당 구간에 속하는 프레임 번호 선택
                interval_frames = [
                    frame for frame in common_frames if start_frame <= frame < end_frame
                ]

                if interval_frames:
                    interval_total_frames = len(interval_frames)
                    interval_matching_count = sum(
                        1 for frame in interval_frames if dict1[str(frame)] == dict2[str(frame)]
                    )
                    matching_rate = interval_matching_count / interval_total_frames
                    # 다음 인덱스를 사용하여 이 구간의 결과를 저장
                    matching_rates[idx + 1] = (matching_rate, start_time, end_time)
                else:
                    print(f"구간 {idx + 1} ({start_time}s ~ 끝)에 해당하는 프레임이 없습니다.")
                    matching_rates[idx + 1] = None

    return matching_rates

intervals = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000,),(1000,1200)]
rates = calculate_matching_rate(label_data_no_segment, selected_channels, intervals, 30)

if rates['total'] is not None:
        print(f"전체 구간의 일치율: {rates['total'] * 100:.2f}%")
else:
    print("전체 구간에 해당하는 프레임이 없습니다.")

for idx in sorted(k for k in rates.keys() if k != 'total'):
    rate_info = rates[idx]
    if rate_info is not None:
        if isinstance(rate_info, tuple):
            # 자동 추가된 마지막 구간
            matching_rate, start_time, end_time = rate_info
            print(f"구간 {idx} ({start_time}s ~ {end_time}): 일치율: {matching_rate * 100:.2f}%")
        else:
            # 지정된 구간
            matching_rate = rate_info
            start_time = intervals[idx][0]
            end_time = intervals[idx][1]
            print(f"구간 {idx} ({start_time}s ~ {end_time}s): 일치율: {matching_rate * 100:.2f}%")
    else:
        print(f"구간 {idx}: 비교할 프레임이 없습니다.")