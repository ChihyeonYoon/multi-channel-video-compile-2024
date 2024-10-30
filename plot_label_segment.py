import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sympy import plot

def plot_inferred_data(infered_data, fps=30, output_path='output_plot.png'):
    """
    infered_data = {str(frame_num):str(speaker), ...}
    """
    
    color_palette = {
        "Interst_Segment": "skyblue",
        "SPEAKER_00": "#264653",
        "SPEAKER_01": "#2A9D8F",
        "SPEAKER_02": "#E9C46A",
        "SPEAKER_03": "#F4A261",
        "UNKNOWN_SPEAKER": "#E76F51"
    }
    
    time_points = [int(frame) / fps for frame in infered_data.keys()]
    speakers = list(infered_data.values())

    # Define speaker order (from bottom to top)
    ordered_speakers = ["SPEAKERS_OVER_TIME", "UNKNOWN_SPEAKER", "SPEAKER_03", "SPEAKER_02", "SPEAKER_01", "SPEAKER_00"]
    speaker_indices = {speaker: idx for idx, speaker in enumerate(ordered_speakers)}
    
    # Define colors based on speakers using color_palette
    colors = [color_palette.get(speaker, "#000000") for speaker in speakers]
    
    # Define plot
    plt.figure(figsize=(15, 5))
    plt.scatter(time_points, [speaker_indices[speaker] for speaker in speakers], c=colors)
    
    # Adding a combined y-value for all speakers at the bottom
    combined_y_value = speaker_indices["SPEAKERS_OVER_TIME"]
    plt.scatter(time_points, [combined_y_value for _ in speakers], c=colors, alpha=0.3)
    
    # Update y-ticks to include the combined value
    plt.yticks(range(len(ordered_speakers)), ordered_speakers)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speakers')
    plt.title('Inferred Data - Speakers Over Time')
    plt.savefig(output_path)

def plot_inferred_data2(infered_data, fps=30, output_path='output_plot2.png'):
    """
    infered_data = {str(frame_num):str(speaker), ...}
    """
    
    color_palette = {
        "Interst_Segment": "skyblue",
        "SPEAKER_00": "#264653",
        "SPEAKER_01": "#2A9D8F",
        "SPEAKER_02": "#E9C46A",
        "SPEAKER_03": "#F4A261",
        "UNKNOWN_SPEAKER": "#E76F51"
    }
    
    time_points = [int(frame) / fps for frame in infered_data.keys()]
    speakers = list(infered_data.values())

    # Define speaker order (from bottom to top)
    ordered_speakers = ["SPEAKERS_OVER_TIME", "UNKNOWN_SPEAKER", "SPEAKER_03", "SPEAKER_02", "SPEAKER_01", "SPEAKER_00"]
    speaker_indices = {speaker: idx for idx, speaker in enumerate(ordered_speakers)}
    
    # Define plot
    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    # Add rectangles for each time segment
    for i in range(len(time_points) - 1):
        start_time = time_points[i]
        end_time = time_points[i + 1]
        speaker = speakers[i]
        y_value = speaker_indices[speaker]
        color = color_palette.get(speaker, "#000000")

        rect = patches.Rectangle((start_time, y_value - 0.4), end_time - start_time, 0.8, color=color, alpha=0.8)
        ax.add_patch(rect)

    # Adding a combined y-value for all speakers at the bottom
    for i in range(len(time_points) - 1):
        start_time = time_points[i]
        end_time = time_points[i + 1]
        color = color_palette.get(speakers[i], "#000000")
        combined_y_value = speaker_indices["SPEAKERS_OVER_TIME"]
        rect = patches.Rectangle((start_time, combined_y_value - 0.4), end_time - start_time, 0.8, color=color, alpha=0.3)
        ax.add_patch(rect)
    
    # Update y-ticks to include the combined value
    plt.yticks(range(len(ordered_speakers)), ordered_speakers)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speakers')
    plt.title('Inferred Data - Speakers Over Time')
    plt.xlim(min(time_points), max(time_points))
    plt.ylim(-1, len(ordered_speakers))
    plt.savefig(output_path)

def plot_inferred_data3(infered_data, segment_data, fps=30, output_path='output_plot.png'):
    """
    infered_data = {str(frame_num):str(speaker), ...}
    segment_data = [
        {
            "start": int,
            "end": int,
            "frames": {str(frame_num):str(speaker), ...}
        },
        ...
    ]
    """
    
    color_palette = {
        "Interst_Segment": "skyblue",
        "SPEAKER_00": "#264653",
        "SPEAKER_01": "#2A9D8F",
        "SPEAKER_02": "#E9C46A",
        "SPEAKER_03": "#F4A261",
        "UNKNOWN_SPEAKER": "#E76F51"
    }
    
    time_points = [int(frame) / fps for frame in infered_data.keys()]
    speakers = list(infered_data.values())

    # Define speaker order (from bottom to top)
    ordered_speakers = ["Interst_Segment", "Interst_SPEAKERS_OVER_TIME", "SPEAKERS_OVER_TIME", "UNKNOWN_SPEAKER", "SPEAKER_03", "SPEAKER_02", "SPEAKER_01", "SPEAKER_00"]
    speaker_indices = {speaker: idx for idx, speaker in enumerate(ordered_speakers)}
    
    # Define plot
    plt.figure(figsize=(15, 7))
    ax = plt.gca()

    # Add rectangles for each time segment from inferred data
    for i in range(len(time_points) - 1):
        start_time = time_points[i]
        end_time = time_points[i + 1]
        speaker = speakers[i]
        y_value = speaker_indices[speaker]
        color = color_palette.get(speaker, "#000000")

        rect = patches.Rectangle((start_time, y_value - 0.4), end_time - start_time, 0.8, color=color, alpha=0.8)
        ax.add_patch(rect)

    # Adding a combined y-value for all speakers at the bottom
    for i in range(len(time_points) - 1):
        start_time = time_points[i]
        end_time = time_points[i + 1]
        color = color_palette.get(speakers[i], "#000000")
        combined_y_value = speaker_indices["SPEAKERS_OVER_TIME"]
        rect = patches.Rectangle((start_time, combined_y_value - 0.4), end_time - start_time, 0.8, color=color, alpha=0.3)
        ax.add_patch(rect)

    # Add rectangles for each segment from segment data
    for segment in segment_data:
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        color = color_palette.get("Interst_Segment", "skyblue")
        
        # Draw the Interst_Segment bar
        interst_segment_y_value = speaker_indices["Interst_Segment"]
        rect = patches.Rectangle((start_time, interst_segment_y_value - 0.4), end_time - start_time, 0.8, color=color, alpha=0.5)
        ax.add_patch(rect)
        
        # Draw the Interst_SPEAKERS_OVER_TIME bar based on frames
        interst_speakers_y_value = speaker_indices["Interst_SPEAKERS_OVER_TIME"]
        for frame_num, speaker in segment["frames"].items():
            frame_time = int(frame_num) / fps
            speaker_color = color_palette.get(speaker, "skyblue")
            rect = patches.Rectangle((frame_time, interst_speakers_y_value - 0.4), 1 / fps, 0.8, color=speaker_color, alpha=0.3)
            ax.add_patch(rect)
    
    # Update y-ticks to include the combined value
    plt.yticks(range(len(ordered_speakers)), ordered_speakers)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speakers')
    plt.title('Inferred Data - Speakers Over Time')
    plt.xlim(min(time_points), max(time_points))
    plt.ylim(-1, len(ordered_speakers))
    plt.savefig(output_path)


if __name__ == '__main__':    
    infered_data = json.load(open('./outputs/selected_channels.json', 'r'))
    """
    infered_data = {str(frame_num):str(speaker), ...}
    """
    
    segment_data = json.load(open('./materials/thelive/annotations_thelive.json', 'r'))
    """
    segment_data = [
        {
            "start": int,
            "end": int,
            "frames": {str(frame_num):str(speaker), ...}
            },
        ...
        ]
    """

    color_palette = {
        "Interst_Segment": "skyblue",
        "SPEAKER_00": "#264653",
        "SPEAKER_01": "#2A9D8F",
        "SPEAKER_02": "#E9C46A",
        "SPEAKER_03": "#F4A261",
        "UNKNOWN_SPEAKER": "#E76F51"
    }
    
    plot_inferred_data(infered_data, output_path='output_plot.png')
    plot_inferred_data2(infered_data, output_path='output_plot2.png')
    plot_inferred_data3(infered_data, segment_data, output_path='output_plot3.png')