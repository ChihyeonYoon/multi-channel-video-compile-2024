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
        model_wight= "/NasData/home/ych/Diar_Recog_Ident/workspace/0813/pytorch_model.bin",
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.model = Model.from_pretrained('/NasData/home/ych/Diar_Recog_Ident/workspace/0813/pytorch_model.bin')
        self.pipeline = SpeakerDiarization(segmentation=self.model)
        HYPER_PARAMETERS = {
            'segmentation': {
                'min_duration_off': 0.5817029604921046,
                # 'threshold': 0.4442333667381752
                },
            'clustering': 
            {
                'method': 'centroid',
                'min_cluster_size': 15,
                'threshold': 0.7153814381597874
                }
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)
        self.pipeline = self.pipeline.to(torch.device(device))
    
    def __call__(self, audio: Union[str, np.ndarray], num_speakers=2):
        segments = self.pipeline(audio, num_speakers=num_speakers)
        print(self.pipeline)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df

device = "cuda" 
audio_file = "/NasData/home/ych/Diar_Recog_Ident/workspace/materials/thelive/audio.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
# diarize_model = whisperx.DiarizationPipeline(use_auth_token="your access token", device=device)
diarize_model = PyannoteDiarizationPipeline()

# add min/max number of speakers if known
# diarize_segments = diarize_model(audio, min_speakers=4, max_speakers=4)
diarize_segments = diarize_model(audio_file, num_speakers=4)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# result = whisperx.assign_word_speakers(diarize_segments, result)
result = whisperx.assign_word_speakers(diarize_segments, result)

with open("transcriptions_.json", "w", encoding='utf8') as f:
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

with open("diarizations_.json", "w", encoding='utf8') as f:
    json.dump(diarizations, f, ensure_ascii=False, indent=4)

