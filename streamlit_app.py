import streamlit as st

import os
import time

from audiorecorder import audiorecorder
from st_copy_to_clipboard import st_copy_to_clipboard as copy

from ASR.loadWhisper import loadWhisper
from Seq2Tex import Seq2Tex_fr


st.title('Wav2Tex experimental model (fr)')

accepted_models = ['base', 'small', 'medium']

if not 'ASR' in st.session_state :   st.session_state.ASR = None 
if not 'Seq2Tex' in st.session_state : st.session_state.Seq2Tex = None

# *****************************************
# *                                       *
# *            WHISPER CONFIG             *
# *                                       *
# *****************************************

st.write("___")
st.subheader(":orange[Configuration]")

def load_model(model_type:str):
    return loadWhisper(model_type, 'fr')

st.write(":red[whisper config]")
c1, c2 = st.columns(2)
with c1 : select_model = st.selectbox("Whisper model size", accepted_models)
with c2 :
    if st.button('Load ASR'):
        st.session_state.ASR = load_model(select_model)

# *****************************************
# *                                       *
# *              FST loading              *
# *                                       *
# *****************************************

st.write(":red[FST config]")

if st.session_state.Seq2Tex == None : st.session_state.Seq2Tex = Seq2Tex_fr()

# *****************************************
# *                                       *
# *                Demo                   *
# *                                       *
# *****************************************

st.write("___")
st.subheader(":orange[Demo]")
  
audio = audiorecorder("Enregistrer", "Terminer l'enregistrement")

if len(audio) > 0:

    # To play audio in frontend:
    st.audio(audio.export().read())  

    # To save audio to a file, use pydub export method:
    audio.export("audio.wav", format="wav")

    # To get audio properties, use pydub AudioSegment properties:
    st.write(f"Sample rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

    if st.session_state.ASR != None :

        wav2seq = st.session_state.ASR.predict('audio.wav')
        seq2tex =  st.session_state.Seq2Tex.predict(wav2seq)
        
        st.write(wav2seq)
        a, b = st.columns(2)
        with a :  
            st.write('**:green[Sortie LaTeX :]**')
        with b :
            d, e = st.columns(2)
            with d :
                st.write('$' + seq2tex + '$')
            with e :
                copy(seq2tex)
                
    else :

        st.write(":red[Download ASR model before]")

