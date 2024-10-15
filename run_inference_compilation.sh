audio_path="materials/thelive/audio.wav"
wide_ch_video="materials/thelive/W.mp4"

# In the current case, MC_left and MC_right are shown in one channel in the original video.
# So we divided the area in half of the original video and separated MC_left and MC_right into two channels so that only one speaker can appear, and used them for inference.
# Also, in the compliation process, the original video with two speakers is input into two channels each representing MC. 
# Therefore, MC_left and MC_right in the compliation process are the same video.
# If you have samples representing one person per channel, 
# it is correct to use the same shell script variable as the input for inference and compilation.
# No distinction, Like this:
#     speaker_ch_videos="materials/spk1.mp4 materials/spk2.mp4 materials/spk3.mp4 materials/spk4.mp4"

speaker_ch_videos_for_inference="materials/thelive/C.mp4 materials/thelive/D.mp4 materials/thelive/MC_left_.mp4 materials/thelive/MC_right_.mp4"
speaker_ch_videos_for_compilation="materials/thelive/C.mp4 materials/thelive/D.mp4 materials/thelive/MC_left.mp4 materials/thelive/MC_right.mp4"
num_speakers=4

start_time=60
end_time=120

diarization_model="./materials/pytorch_model.bin"
face_inference_model="./materials/speaking_detection_model_weight.pth"

save_path="./outputs"

# Audio inference
read -p "Do you want to continue to audio inference? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "Start to audio inference."
    python trans_diar_whisperx_pyan.py --audio_path $audio_path --num_speakers $num_speakers \
                                   --diarization_model $diarization_model \
                                   --save_path $save_path \

    sleep 1s
    echo "Inference is done."
else
    echo "You chose not to continue."
    # exit 1
fi

# Visual inference
read -p "Do you want to continue to visual inference? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "You chose to continue."
    python multi_channel_face_infer.py --wide_ch_video $wide_ch_video \
                                   --speaker_ch_videos $speaker_ch_videos_for_inference \
                                   --face_inference_model $face_inference_model \
                                   --save_path $save_path

    sleep 1s
    echo "Inference is done."
else
    echo "You chose not to continue."
    # exit 1
fi

# Compilation
read -p "Do you want to continue to compile final video? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    if [ -f $save_path/transcription.json ] && [ -f $save_path/multi_channel_face_infer.json ]; then
    echo "Start to compile the final video."
    python compile_final_video_copy.py --wide_ch_video $wide_ch_video \
                                --speaker_ch_videos $speaker_ch_videos_for_compilation \
                                --audio_path $audio_path \
                                --start_time $start_time \
                                --end_time $end_time \
                                --transcript_path $save_path/transcription.json \
                                --channel_inference_file $save_path/multi_channel_face_infer.json \
                                --save_path $save_path
    else
        echo "Something went wrong in the inference process."
    fi
else
    echo "You chose not to continue."
fi





