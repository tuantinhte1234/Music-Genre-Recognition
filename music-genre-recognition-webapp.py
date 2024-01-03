import streamlit as st
import base64
import pytube
import os
import subprocess 
import librosa

st.set_page_config(layout="wide", page_title="Music Genre Recognition App")
st.write("## Know the genre of your favorite musics!")

st.sidebar.write("## Upload the youtube url of a music of your choice:")
#st.sidebar.image("images_webapp/icons8-youtube.gif")

youtube_url = st.sidebar.text_input('',
                                    placeholder="Paste here the url...",
                                    key="youtube_url",
                                    label_visibility="collapsed",
                                    )


def extract_youtube_mp3(url):
    video = pytube.YouTube(url).streams.filter(only_audio=True).first() 
    mp4_audio = video.download() 
    base, ext = os.path.splitext(mp4_audio) 
    mp3_audio = base + '.mp3'
    os.rename(out_file, mp3_audio) 
    return mp3_audio

def mp3_to_wav(mp3_audio):
    subprocess.call(['ffmpeg', '-i', mp3_audio, os.path.splitext(mp3_audio)[0] + '.wav'])
    wav_audio, sr = librosa.load('converted_to_wav_file.wav')
    return wav_audio, sr

if youtube_url != "": 
    mp3_audio = extract_youtube_mp3(youtube_url)
    wav_audio, sr = mp3_to_wav(mp3_audio)
    st.sidebar.audio(wav_audio, sample_rate=sr)