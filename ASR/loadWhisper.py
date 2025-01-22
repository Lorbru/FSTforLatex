from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
from torchaudio.transforms import Resample
import json
import librosa

sampling_rate = 16000

with open('Tokens/letters.json', 'r', encoding='utf-8') as reader : 
    letters = json.load(reader)
    accepted_chars = letters['tokens']

class loadWhisper():

    ACCEPT_CHARS = accepted_chars
    
    def __init__(self, mod="medium", lang=None):
        self.__processor = WhisperProcessor.from_pretrained("openai/whisper-" + mod, language=lang)
        self.__model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-"+mod)
        

    def predict(self, audio):

        if type(audio) == str : 
            audio, sample_rate = torchaudio.load(audio)
        else : 
            audio, sample_rate = audio
            audio = torch.tensor(audio)

        if audio.dim() == 2 : audio = torch.mean(audio, dim=0) # stereo
        elif audio.dim() != 1 : raise Exception("Error with audio dimension")
       
        resampler = Resample(orig_freq=sample_rate, new_freq=16000) 
        audio = resampler(audio).squeeze()

        # Process audio
        input_features = self.__processor(audio, sampling_rate=16000, truncated=True, return_tensors="pt").input_features

        # Model inference
        predicted_ids = self.__model.generate(input_features)

        # Decoder
        transcription = self.__processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
        
        # keeping chars which will be used for FST
        transcription = ''.join([c for c in transcription if c in accepted_chars or c == ' '])
        
        return transcription