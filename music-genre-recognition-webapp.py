import streamlit as st
import base64

st.set_page_config(layout="wide", page_title="Music Genre Recognition App")
st.write("## Know the genre of your favorite musics!")

st.sidebar.write("## Upload youtube music link")
#st.sidebar.image("images_webapp/icons8-youtube.gif")

youtube_url = st.sidebar.text_input(,
                                    "Paste here the music youtube link",
                                    key="youtube_url",
                                    )



#<a  href="https://icons8.com/icon/37326/youtube">YouTube</a> icon by <a href="https://icons8.com">Icons8</a>