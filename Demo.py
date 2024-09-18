import os
import sys
import streamlit as st
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)
from speech_service import synthesize_speech_audio 
from flow_generator import generate_sequence_text

st.title('AI Yoga Teacher')

recorded_file = 'output_sample.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file)


generated_flow = generate_sequence_text()
print(generated_flow)
flowText = 'Welcome to your yoga flow. We begin with '


poses_audio = []
for pose in generated_flow:
    pose_audio = synthesize_speech_audio(pose)
    poses_audio.append(pose_audio)
    # time.sleep(.10)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Generate Flow', on_click=click_button)

def stream_data(text):
    for word in text.split(" "):
        time.sleep(0.1)
        yield word + " "
        time.sleep(0.2)


placeholder = st.empty()
  
if st.session_state.clicked:

    for i, pose in enumerate(poses_audio):
        with placeholder.container():
            st.audio(pose, autoplay=True)
            st.write_stream(stream_data(generated_flow[i]))
        # Clear all those elements:
        placeholder.empty()

