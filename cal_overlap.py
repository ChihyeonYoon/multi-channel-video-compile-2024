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
    Calculates the matching rates between two dictionaries over specified intervals,
    including any remaining intervals and the total matching rate.

    Parameters:
    dict1 (dict): First dictionary with frame numbers (as strings) as keys and values to compare.
    dict2 (dict): Second dictionary with frame numbers (as strings) as keys and values to compare.
    intervals (list of tuples): List of intervals in the form [(start_time, end_time), ...].
    frame_rate (float): Frames per second (FPS) to convert time to frame numbers.

    Returns:
    dict: Dictionary containing matching rates for each interval and the total matching rate.
    """
    matching_rates = {}

    # Convert frame numbers to integers
    frames1 = set(int(k) for k in dict1.keys())
    frames2 = set(int(k) for k in dict2.keys())

    # Get common frames
    common_frames = frames1.intersection(frames2)

    if not common_frames:
        matching_rates['total'] = None
        print("No common frames between the two dictionaries.")
        return matching_rates

    # Compute total matching rate
    total_frames = len(common_frames)
    matching_count = sum(
        1 for frame in common_frames if dict1[str(frame)] == dict2[str(frame)]
    )
    matching_rates['total'] = matching_count / total_frames

    # Get min and max frame numbers
    min_frame = min(common_frames)
    max_frame = max(common_frames)

    # Process specified intervals
    for idx, (start_time, end_time) in enumerate(intervals):
        # Convert times to frame numbers
        start_frame = int(start_time * frame_rate)
        if end_time == 'end':
            end_frame = max_frame + 1  # Include last frame
        else:
            end_frame = int(end_time * frame_rate)

        # Get frames in interval
        interval_frames = [
            frame for frame in common_frames if start_frame <= frame < end_frame
        ]

        if not interval_frames:
            print(f"No frames in interval {idx} ({start_time}s to {end_time}s).")
            matching_rates[idx] = None
            continue

        interval_total_frames = len(interval_frames)
        interval_matching_count = sum(
            1 for frame in interval_frames if dict1[str(frame)] == dict2[str(frame)]
        )
        matching_rate = interval_matching_count / interval_total_frames
        matching_rates[idx] = matching_rate

    # Check for remaining interval after the last specified interval
    last_end_time = intervals[-1][1]
    if last_end_time != 'end':
        if isinstance(last_end_time, (int, float)):
            last_end_frame = int(last_end_time * frame_rate)
            if last_end_frame < max_frame:
                # There are frames after the last specified interval
                start_time = last_end_time
                end_time = 'end'
                start_frame = last_end_frame
                end_frame = max_frame + 1  # Include last frame

                # Get frames in interval
                interval_frames = [
                    frame for frame in common_frames if start_frame <= frame < end_frame
                ]

                if interval_frames:
                    interval_total_frames = len(interval_frames)
                    interval_matching_count = sum(
                        1 for frame in interval_frames if dict1[str(frame)] == dict2[str(frame)]
                    )
                    matching_rate = interval_matching_count / interval_total_frames
                    # Use next index for this interval
                    matching_rates[idx + 1] = matching_rate
                else:
                    print(f"No frames in interval {idx + 1} ({start_time}s to end).")
                    matching_rates[idx + 1] = None

    return matching_rates

intervals = [(0, 10), (10, 150)]
rates = calculate_matching_rate(label_data_no_segment, selected_channels, intervals, 30)

if rates['total'] is not None:
        print(f"Total matching rate: {rates['total'] * 100:.2f}%")
else:
    print("No common frames in total interval.")

for idx in sorted(k for k in rates.keys() if k != 'total'):
    rate = rates[idx]
    if rate is not None:
        if idx == len(intervals):
            print(f"Interval {idx} (150s to end): Matching rate: {rate * 100:.2f}%")
        else:
            start_time = intervals[idx][0]
            end_time = intervals[idx][1]
            print(f"Interval {idx} ({start_time}s to {end_time}s): Matching rate: {rate * 100:.2f}%")
    else:
        print(f"Interval {idx}: No frames to compare.")

