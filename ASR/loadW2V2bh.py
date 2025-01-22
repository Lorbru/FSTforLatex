import torch
import torchaudio
import json

from transformers import AutoModelForCTC, Wav2Vec2Processor

with open('Tokens/letters.json', 'r', encoding='utf-8') as reader : 
    letters = json.load(reader)
    accepted_chars = letters['tokens']

class loadW2V2bh():

    ACCEPT_CHARS = accepted_chars

    def __init__(self):

        self.device = "cpu"
        self.model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
        self.model_sample_rate = self.processor.feature_extractor.sampling_rate

        

    def predict(self, audio):

        if type(audio) == str : 
            waveform, sample_rate = torchaudio.load(audio)
        else : 
            waveform, sample_rate = audio
            waveform = torch.tensor(waveform)

        if waveform.dim() == 2 : waveform = torch.mean(waveform, dim=0) # stereo
        elif waveform.dim() != 1 : raise Exception("Error with audio dimension")

        # resample
        if sample_rate != self.model_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.model_sample_rate)
            waveform = resampler(waveform)

        # normalize
        input_dict = self.processor(waveform, sampling_rate=self.model_sample_rate, return_tensors="pt")

        with torch.inference_mode():
            logits = self.model(input_dict.input_values.to(self.device)).logits

        # decode
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0]