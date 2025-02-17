import pandas as pd
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings('ignore')

from ASR.loadW2V2bh import *
from ASR.loadWhisper import *
from Datasets.Audiodataset import *
from Scores.Metrics import *
from Seq2Tex import *
from Datasets.Fleurs import FleursdatasetTest

def get_audio_duration(file_path): 
    info = torchaudio.info(file_path) 
    duration = info.num_frames / info.sample_rate 
    return duration


def Wav2Vec2Analysis():

    print('  * Data loading')
    dataset = Audiodata()


    print(f'  * W2V2bh loading --')
    model = loadW2V2bh()
    
    WER = []
    RTF = []
    pred = []
    pids = []
    dnas = []
    files = []

    print('  * Normalization')
    normalizer = Normalemm()

    print('  * Score analysis')
    for i in tqdm(range(len(dataset))):

        # sample
        sample = dataset[i]

        dna = sample['dna']
        pid = sample['pid']
        file = sample['audio']
        files.append(file.split('.wav')[0].split(audio_path)[1])

        # True sequence
        Ytrue = normalizer.predict(sample['sequence'])

        # audio duration
        duration = get_audio_duration(file)

        # whisper prediction + normalization
        t0 = time.time()
        Ypred = model.predict(file)
        Ypred = normalizer.predict(Ypred)

        RTF.append( (time.time() - t0) / duration )
        WER.append(WordErrorRate(Ytrue, Ypred))
        pred.append(Ypred)
        pids.append(pid)
        dnas.append(dna)

    pd.DataFrame({
            'file_id':files,
            'wer':WER,
            'rtf':RTF,
            'pid':pids,
            'dna':dnas,
            'pred':pred
        }).to_csv(f'Scores/score_w2v2bh.csv', index=False)

def Wav2Vec2bhFleursAnalysis():

    print('  * Data loading')
    dataset = FleursdatasetTest()

    print(f'  * w2v2 loading ')
    model = loadW2V2bh()
    
    WER = []
    RTF = []
    pred = []

    print('  * Normalization')
    normalizer = Normalemm()

    print('  * Score analysis')
    for i in tqdm(range(len(dataset))):
    #for i in tqdm(range(5)):

        # sample
        audio, seq, sample_rate = dataset[i]

        # True sequence
        seq = ''.join([c for c in seq if c in loadW2V2bh.ACCEPT_CHARS or c == ' '])
        Ytrue = normalizer.predict(seq)

        # audio duration
        duration = len(audio)/sample_rate

        # whisper prediction + normalization
        t0 = time.time()
        Ypred = model.predict((audio, sample_rate))
        Ypred = normalizer.predict(Ypred)

        RTF.append( (time.time() - t0) / duration )
        WER.append(WordErrorRate(Ytrue, Ypred))
        pred.append(Ypred)

    pd.DataFrame({
            'wer':WER,
            'rtf':RTF,
        }).to_csv(f'Scores/score_w2v2_fleurs.csv', index=False)

def WFSTAnalysis():

    print('  * Data loading')
    dataset = Audiodata()
    
    WER = []
    WER10 = []
    RTF = []
    pids = []
    dnas = []
    files = []

    print('  * Seq2Tex WFST')
    model = Seq2Tex_fr()

    print('  * Score analysis')
    
    for i in tqdm(range(len(dataset))):

        # sample
        sample = dataset[i]

        dna = sample['dna']
        pid = sample['pid']
        file = sample['audio']
        files.append(file.split('.wav')[0].split(audio_path)[1])

        # True sequence
        Ytrue = sample['latex']
        # audio duration
        duration = get_audio_duration(file)

        t0 = time.time()
        Ypred = model.predict(sample['sequence'])

        RTF.append( (time.time() - t0) / duration )
        WER.append(WordErrorRate(Ytrue, Ypred))
        
        best_wer = WER[-1]
        best_hyp = Ypred
        for Hyp in model.outputs(sample['sequence'], 10):
            wer = WordErrorRate(Ytrue, Hyp)
            if wer < best_wer  :
                best_wer = wer

        WER10.append(best_wer)
        pids.append(pid)
        dnas.append(dna)

    pd.DataFrame({
            'file_id':files,
            'wer':WER,
            'wer10':WER10,
            'rtf':RTF,
            'pid':pids,
            'dna':dnas,
        }).to_csv(f'Scores/score_fst_lex_gram.csv', index=False)
    
    ### only lexical part ###

    WER = []
    RTF = []

    print('  * Lex WFST')
    model = Normalemmlex()

    print('  * Score analysis')
    for i in tqdm(range(len(dataset))):

        # sample
        sample = dataset[i]

        # True sequence
        Ytrue = sample['latex']

        # audio duration
        duration = get_audio_duration(file)

        t0 = time.time()
        Ypred = model.predict(sample['sequence'])

        RTF.append( (time.time() - t0) / duration )
        WER.append(WordErrorRate(Ytrue, Ypred))

    pd.DataFrame({
            'file_id':files,
            'wer':WER,
            'rtf':RTF,
            'pid':pids,
            'dna':dnas,
        }).to_csv(f'Scores/score_fst_lex.csv', index=False)
    
def WFSTASRAnalysis():

    csvs = [
        "Scores/score_whisper_base.csv",
        "Scores/score_whisper_small.csv",
        "Scores/score_whisper_medium.csv",
        "Scores/score_w2v2bh.csv"
    ]

    labels = [
        "base",
        "small",
        "medium",
        "w2v2bh"
    ]

    print('  * Data loading')
    dataset = Audiodata()
    
    print('  * Seq2Tex WFST')
    model = Seq2Tex_fr()

    print('  * Score analysis')
    for t, csv in enumerate(csvs) : 
        wb = pd.read_csv(csv)
        WER = []
        WER10 = []
        pids = []
        dnas = []

        for i in tqdm(range(len(dataset))):
        # for i in tqdm(range(3)):

            # sample
            sample = dataset[i]

            dna = sample['dna']
            pid = sample['pid']
            file = sample['audio']

            # True sequence
            Ytrue = sample['latex']
            try :
                pred = str(wb.loc[wb['file_id'] == file.split('.wav')[0].split(audio_path)[1]]['pred'].item())
            except :
                print(pred)
                pred = ''
            pred = ''.join([c for c in pred if c in accepted_chars or c==' '])
            Ypred = model.predict(pred)

            WER.append(WordErrorRate(Ytrue, Ypred))
            
            # best_wer = WER[-1]
            # for Hyp in model.outputs(pred, 10):
            #    wer = WordErrorRate(Ytrue, Hyp)
            #    if wer < best_wer  :
            #        best_wer = wer

            #WER10.append(best_wer)
            pids.append(pid)
            dnas.append(dna)

        pd.DataFrame({
                'wer':WER,
                 # 'wer10':WER10,
                'pid':pids,
                'dna':dnas,
            }).to_csv(f'Scores/score_asr_{labels[t]}_fst_lex_gram.csv', index=False)
    
    ### only lexical part ###

    

    print('  * Lex WFST')
    model = Normalemmlex()

    print('  * Score analysis')
    for t, csv in enumerate(csvs) : 
        wb = pd.read_csv(csv)
        
        WER = []
        pids = []
        dnas = []

        for i in tqdm(range(len(dataset))):
        # for i in tqdm(range(3)):

            # sample
            sample = dataset[i]

            # True sequence
            Ytrue = sample['latex']

            dna = sample['dna']
            pid = sample['pid']
            file = sample['audio']

            pred = wb.loc[wb['file_id'] == file.split('.wav')[0].split(audio_path)[1]]['pred'].item()
            try :
                pred = str(wb.loc[wb['file_id'] == file.split('.wav')[0].split(audio_path)[1]]['pred'].item())
            except :
                pred = ''
            Ypred = model.predict(pred)
            WER.append(WordErrorRate(Ytrue, Ypred))
            pids.append(pid)
            dnas.append(dna)

        pd.DataFrame({
                'wer':WER,
                'pid':pids,
                'dna':dnas,
            }).to_csv(f'Scores/score_asr_{labels[t]}_fst_lex.csv', index=False)

def WhisperAnalysis():

    print('  * Data loading')
    dataset = Audiodata()

    for model_size in ['base', 'small', 'medium']:

        print(f'  * Whisper loading -- {model_size}')
        model = loadWhisper(model_size, lang='fr')
        
        WER = []
        RTF = []
        pred = []
        pids = []
        dnas = []
        files = []

        print('  * Normalization')
        normalizer = Normalemm()

        print('  * Score analysis')
        for i in tqdm(range(len(dataset))):

            # sample
            sample = dataset[i]

            dna = sample['dna']
            pid = sample['pid']
            file = sample['audio']
            files.append(file.split('.wav')[0].split(audio_path)[1])

            # True sequence
            Ytrue = normalizer.predict(sample['sequence'])

            # audio duration
            duration = get_audio_duration(file)

            # whisper prediction + normalization
            t0 = time.time()
            Ypred = model.predict(file)
            Ypred = normalizer.predict(Ypred)

            RTF.append( (time.time() - t0) / duration )
            WER.append(WordErrorRate(Ytrue, Ypred))
            pred.append(Ypred)
            pids.append(pid)
            dnas.append(dna)

        pd.DataFrame({
                'file_id':files,
                'wer':WER,
                'rtf':RTF,
                'pid':pids,
                'dna':dnas,
                'pred':pred
            }).to_csv(f'Scores/score_whisper_{model_size}.csv', index=False)

def WhisperFleursAnalysis():

    print('  * Data loading')
    dataset = FleursdatasetTest()

    for model_size in ['base', 'small', 'medium']:

        print(f'  * Whisper loading -- {model_size}')
        model = loadWhisper(model_size, lang='fr')
        
        WER = []
        RTF = []
        pred = []

        print('  * Normalization')
        normalizer = Normalemm()

        print('  * Score analysis')
        for i in tqdm(range(len(dataset))):
        # for i in tqdm(range(5)):

            # sample
            audio, seq, sample_rate = dataset[i]

            # True sequence
            seq = ''.join([c for c in seq if c in loadWhisper.ACCEPT_CHARS or c == ' '])
            Ytrue = normalizer.predict(seq)

            # audio duration
            duration = len(audio)/sample_rate

            # whisper prediction + normalization
            t0 = time.time()
            Ypred = model.predict((audio, sample_rate))
            Ypred = normalizer.predict(Ypred)

            RTF.append( (time.time() - t0) / duration )
            WER.append(WordErrorRate(Ytrue, Ypred))
            pred.append(Ypred)

        pd.DataFrame({
                'wer':WER,
                'rtf':RTF,
                'pred':pred
            }).to_csv(f'Scores/score_whisper_fleurs_{model_size}.csv', index=False)

if __name__ == '__main__':
    
    WhisperAnalysis()           # Whisper on Math   Dataset   (base + small + medium)
    WhisperFleursAnalysis()     # Whisper on Fleurs Dataset   (base + small + medium)

    Wav2Vec2Analysis()          # Wav2Vec2 fine tuned fr on Math   Dataset
    Wav2Vec2bhFleursAnalysis()  # Wav2Vec2 fine tuned fr on Fleurs Dataset

    WFSTASRAnalysis()           # ASR (Whisper medium) + WFST Word error rate
    WFSTAnalysis()              # WFST Word error rate from exact retranscription



    

