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