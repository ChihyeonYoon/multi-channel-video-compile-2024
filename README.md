# Multi-Channel Video Compilation

This is the multi-channel video compile solution.
Compile multiple video files into a single video file with multiple channels.
This solution is implemented referring to the whisperx and pyannote-audio speech recognition and speaker diarization solutions.
![Image description](figs/overall_fig.png)
# Installation
pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# OR
pip install -r requirements.txt
```

libcudnn8 is required to run the solution. 

pyanoote.audio == 3.3.1 is required to run the solution.

# Directory Structure
```
|-- ./
    |-- README.md
    |-- requirements.txt
    |-- run_inference_compilation.sh
    |-- trans_diar_whisperx_pyan.py
    |-- multi_channel_face_infer.py
    |-- face_utils.py
    |-- compile_final_video.py
    |-- outputs/
        |-- speaker_segments.png
        |-- speaker_segments2.png
        |-- transcriptions.json
        |-- sample_with_audio.mp4
        |-- multi_channel_face_infer.json
        |-- diarizations.json
        |-- sample.mp4
    |-- materials/
        |   # The following files are not included in the repository
        |   
        |-- pytorch_model.bin
        |-- speaking_detection_model_weight.pth
        |-- thelive/
            |-- D.mp4
            |-- MC_right.mp4
            |-- output_video_left.mp4
            |-- MC_right_.mp4
            |-- W.mp4
            |-- C.mp4
            |-- output_video_right.mp4
            |-- MC.mp4
            |-- audio.wav
            |-- MC_left_.mp4
            |-- MC_left.mp4
            |-- transcript_thelive.json
        |-- opentalk/
            |-- camera3_synced.mp4
            |-- audio_segment.wav
            |-- audio.mp3
            |-- camera2_synced.mp4
            |-- audio.wav
            |-- camera1_synced.mp4
```
You can download pytorch_model.bin from https://huggingface.co/pyannote/segmentation/tree/main
You can download speaking_detection_model_weight.pth from https://drive.google.com/file/d/1dia_na1ci_B1fDfPX5fpJBbofDUvBF1L/view?usp=drive_link
# Visualize the result

The following images are the visualization of the speaker diarization result.
![Image description](./figs/speaker_segments.png)
![Image description](./figs/speaker_segments2.png)

