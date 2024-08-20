"""
Get important line or segmet from Transcript with the lm model
"""

import transformers
import torch
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('upskyy/ko-reranker')
model = AutoModelForSequenceClassification.from_pretrained('upskyy/ko-reranker').to(device)
model.eval()

with open ("transcribe2.json", "r", encoding='utf-8') as f:
    json_file = json.load(f)
    print(type(json_file))

transcripts = []
for item in json_file:
    start = item['start']
    end = item['end']
    text = item['text']
    speaker = item['speaker']
    transcripts.append(text)

all_transcripts = "\n".join(transcripts)
print(len(all_transcripts))

'''
RuntimeError: The expanded size of the tensor (4489) must match the existing size (514) at non-singleton dimension 1.  Target sizes: [1, 4489].  Tensor sizes: [1, 514]
'''

with torch.no_grad():
    for transcript in transcripts:
        inputs = tokenizer([[all_transcripts, transcript]], padding=True, truncation=False, return_tensors='pt',)
        print(inputs)
        inputs = inputs.to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # scores = exp_normalize(scores.cpu().numpy())
        print(scores)


