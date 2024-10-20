audio_path="materials/thelive/audio.wav"
wide_ch_video="materials/thelive/W.mp4"

speaker_ch_videos_for_inference="materials/thelive/C.mp4 materials/thelive/D.mp4 materials/thelive/MC_left_.mp4 materials/thelive/MC_right_.mp4"
speaker_ch_videos_for_compilation="materials/thelive/C.mp4 materials/thelive/D.mp4 materials/thelive/MC_left.mp4 materials/thelive/MC_right.mp4"

num_speakers=4
start_time=0
end_time=9999999999

diarization_model="./materials/pytorch_model.bin"
face_inference_model="./materials/speech_state_estimation_model.pth"

save_path="./outputs"

# 전체 실행 시작 시간 기록
total_start_time=$(date +%s)

# Audio inference
read -p "Do you want to continue to audio inference? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "Start to audio inference."
    audio_start_time=$(date +%s)  # 시작 시간 기록
    
    python trans_diar_whisperx_pyan.py --audio_path $audio_path --num_speakers $num_speakers \
                                   --diarization_model $diarization_model \
                                   --save_path $save_path

    audio_end_time=$(date +%s)  # 종료 시간 기록
    audio_duration=$((audio_end_time - audio_start_time))
    echo "Audio inference took $audio_duration seconds."

    sleep 1s
    echo "Audio inference is done."
else
    echo "You chose not to continue."
    # exit 1
fi

# Visual inference
read -p "Do you want to continue to visual inference? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "Start to visual inference."
    visual_start_time=$(date +%s)  # 시작 시간 기록
    
    python multi_channel_face_infer.py --wide_ch_video $wide_ch_video \
                                   --speaker_ch_videos $speaker_ch_videos_for_inference \
                                   --face_inference_model $face_inference_model \
                                   --save_path $save_path

    visual_end_time=$(date +%s)  # 종료 시간 기록
    visual_duration=$((visual_end_time - visual_start_time))
    echo "Visual inference took $visual_duration seconds."

    sleep 1s
    echo "Visual inference is done."
else
    echo "You chose not to continue."
    # exit 1
fi

# Compilation
read -p "Do you want to continue to compile final video? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    if [ -f $save_path/transcriptions.json ] && [ -f $save_path/multi_channel_face_infer.json ]; then
        echo "Start to compile the final video."
        compile_start_time=$(date +%s)  # 시작 시간 기록

        python compile_final_video.py --wide_ch_video $wide_ch_video \
                                    --speaker_ch_videos $speaker_ch_videos_for_compilation \
                                    --audio_path $audio_path \
                                    --start_time $start_time \
                                    --end_time $end_time \
                                    --transcript_path $save_path/transcriptions.json \
                                    --channel_inference_file $save_path/multi_channel_face_infer.json \
                                    --save_path $save_path

        compile_end_time=$(date +%s)  # 종료 시간 기록
        compile_duration=$((compile_end_time - compile_start_time))
        echo "Compilation took $compile_duration seconds."
    else
        echo "Something went wrong in the inference process."
    fi
else
    echo "You chose not to continue."
fi

# 전체 실행 종료 시간 기록 및 출력
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
echo "Total execution time: $total_duration seconds."
