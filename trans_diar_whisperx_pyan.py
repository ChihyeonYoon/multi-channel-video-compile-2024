import whisperx
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import SpeakerDiarization
import gc 
import json
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import argparse

class PyannoteDiarizationPipeline:
    def __init__(
        self,
        model_wight= './materials/pytorch_model.bin',
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.model = Model.from_pretrained(model_wight)
        self.pipeline = SpeakerDiarization(segmentation=self.model)
        HYPER_PARAMETERS = {
            'segmentation': {
                'min_duration_off': 0.5
                },
            'clustering': 
            {
                'method': 'centroid',
                'min_cluster_size': 15,
                'threshold': 0.5
                }
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)
        self.pipeline = self.pipeline.to(torch.device(device))
    
    def __call__(self, audio: Union[str, np.ndarray], num_speakers=2):
        segments = self.pipeline(audio, num_speakers=num_speakers)
        # print(self.pipeline)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="./materials/thelive/audio.wav")
    parser.add_argument("--num_speakers", type=int, default=4)

    parser.add_argument("--diarization_model", type=str, default='./materials/pytorch_model.bin')
    parser.add_argument("--save_path", type=str, default='./compiled_smaple')
    args = parser.parse_args()

    device = "cuda" 
    audio_file = args.audio_path
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels
    diarize_pipeline = PyannoteDiarizationPipeline(model_wight=args.diarization_model, device=device)
    diarize_segments = diarize_pipeline(audio_file, num_speakers=args.num_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    with open(args.save_path + "/transcriptions.json", "w", encoding='utf8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=4)

    diarizations=[]
    for i in range(len(diarize_segments)):
        item = {
            "segment": (diarize_segments.iloc[i].loc['segment'].start, diarize_segments.iloc[i].loc['segment'].end),
            "start": diarize_segments.iloc[i].loc['start'],
            "end": diarize_segments.iloc[i].loc['end'],
            "speaker": diarize_segments.iloc[i].loc['speaker'],
            'intersection': diarize_segments.iloc[i].loc['intersection'],
            'union': diarize_segments.iloc[i].loc['union']
        }
        diarizations.append(item)

    with open(args.save_path + "/diarizations.json", "w", encoding='utf8') as f:
        json.dump(diarizations, f, ensure_ascii=False, indent=4)

