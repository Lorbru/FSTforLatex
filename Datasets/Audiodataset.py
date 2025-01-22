import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

audio_path = 'Data/audio/'
latex_path = 'Data/latex/'
sequence_path = 'Data/sequences/'

class Audiodata(Dataset):

    """
    -- Loading our test dataset
    """
    def __init__(self):
        
        self.samples = []
        self._csv_texseq = pd.read_csv('Data/data_analysis.csv')[['audio_path', 'tex']]
        
        for filename in os.listdir(audio_path):
            
            if filename.endswith('.wav'):
                
                pid = 1 if 'p1' in filename else 2 if 'p2' in filename else 3
                has_dna = 'dna' in filename
                
                audio = os.path.join(audio_path, filename)
                sequence = os.path.join(sequence_path, filename.replace('.wav', '.txt'))
                latex = os.path.join(latex_path, filename.replace('.wav', '.txt'))
                
                self.samples.append((audio, sequence, latex, pid, has_dna))
        

    def __len__(self):
        """
        -- len(dataset)
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        -- get an item 
        """
        audio_file, sequence_file, latex_file, pid, dna = self.samples[idx]

        with open(sequence_file, 'r', encoding='utf-8') as reader: sequence = reader.read().split('\n')[0].strip()

        # with open(latex_file, 'r',  encoding='utf-8') as reader: latex = reader.read().split('\n')[0].strip()
        latex = self._csv_texseq.loc[self._csv_texseq['audio_path'] == audio_file]['tex'].item()

        return {

            'audio': audio_file,                 # audio file
            'sequence': sequence,                # sequence (natural language)
            'latex': latex,                      # latex (latex sequence)
            'pid': pid,                          # speaker id
            'dna': dna,                          # dna == True : natural language/dna == False : constraints for the speaker

        }