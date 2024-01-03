import streamlit as st
import base64
import pytube
import os
import subprocess 
import librosa
import tempfile 
from pydub import AudioSegment
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img,img_to_array


#begin seeting up webapp title and background

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1530176928500-2372a88e00b5?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
#st.title("Music Genre Recognition App")
st.markdown("<h1 style='text-align: center; color: black;'>Music Genre Recognition App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Know the genre of your favorite musics! </h2>", unsafe_allow_html=True)  

#define social medias
col1,col2 = st.sidebar.columns(2)

linked_in = '''[![LinkedIn](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/LinkedIn_Logo.svg/120px-LinkedIn_Logo.svg.png)](https://www.linkedin.com/in/david-iss%C3%A1/)'''
col1.markdown(linked_in, unsafe_allow_html=True)
github = '''[![LinkedIn](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/GitHub_logo_2013.svg/120px-GitHub_logo_2013.svg.png)](https://github.com/davidissa99)'''
col2.markdown(github, unsafe_allow_html=True)


# upload mp3 file and visualize it
st.sidebar.write("## Upload the mp3 file of a music of your choice:")
mp3_file = st.sidebar.file_uploader("Upload an audio file", type=["mp3"], label_visibility="collapsed")


#define function to convert mp3 to wav format
def convert_mp3_to_wav(music_file):  
    sound = AudioSegment.from_mp3(music_file)
    sound.export("music_file.wav",format="wav")
  
#define funciton to produce and save mel spectogram
def create_melspectrogram(wav_file):  
    y,sr = librosa.load(wav_file)  
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr))    
    plt.figure(figsize=(10, 5))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel_spec, x_axis="time", y_axis='mel', sr=sr)
    plt.margins(0)
    plt.savefig('melspectrogram.png')

#define function to predict music genre based on mel spectogram
def predict(image_data, model):   
    image = img_to_array(image_data)   
    image = np.reshape(image,(1,100,200,4))   
    prediction = model.predict(image/255)   
    prediction = prediction.reshape((10,))     
    class_label = np.argmax(prediction)     
    return class_label, prediction

class_labels = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'jazz', 'metal', 'reggae', 'rock']
    

#convert mp3 file to wav and listen to it
if mp3_file is not None:    
  st.sidebar.write("**Play the song below if you want!**")
  st.sidebar.audio(mp3_file,"audio/mp3")
    
  convert_mp3_to_wav(mp3_file)
  
  create_melspectrogram("music_file.wav")
  image_data = load_img('melspectrogram.png', color_mode='rgba', target_size=(100,200))   
        
        
    