import av
import os
import sys
import time
import streamlit as st
import threading
from streamlit_webrtc import WebRtcMode, VideoHTMLAttributes, webrtc_streamer

# import queue
# import urllib.request
# from collections import deque
# from pathlib import Path
# from typing import List

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro
from speech_service import synthesize_speech_audio, speech_synthesis_to_audio_data_stream 
from flow_generator import generate_sequence_text

st.title('AI Yoga Teacher')

lock = threading.Lock()

generated_flow = generate_sequence_text()

flow = " ".join(generated_flow)
flow_audio = speech_synthesis_to_audio_data_stream(flow)

# poses_audio = []
# for pose in generated_flow:
#     pose_audio = speech_synthesis_to_audio_data_stream(pose)
#     poses_audio.append(pose_audio)
#     time.sleep(.10)

# if mode == 'Beginner':
thresholds = get_thresholds_beginner()

live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
# Initialize face mesh solution
pose = get_mediapipe_pose()


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="rgb24")  # Decode and get RGB frame
    frame, _ = live_process_frame.process(frame, pose)  # Process frame
    return av.VideoFrame.from_ndarray(frame, format="rgb24")  # Encode and return BGR frame


left, right = st.columns([0.7, 0.3], gap = 'large')

with left:
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
        media_stream_constraints={"video": True, "audio": False},
    )

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

def stream_data(text):
    for word in text.split(" "):
        time.sleep(0.1)
        yield word + " "
        time.sleep(0.2)

with right:
    st.button('Generate Flow', on_click=click_button)

    with lock:
        placeholder = st.empty()   
        if st.session_state.clicked:
            # for i, pose in enumerate(poses_audio):
            with placeholder.container():
                st.audio(flow_audio, autoplay=True)
                st.write_stream(stream_data(flow))
            time.sleep(1)
            # Clear all those elements:
            placeholder.empty()


# def app_sst_with_video(
    
# ):
#     text = generated_flow[0]
#     frames_deque_lock = threading.Lock()
#     frames_deque: deque = deque([])

#     async def queued_audio_frames_callback(
#         frames: List[av.AudioFrame],
#     ) -> av.AudioFrame:
#         frames = poses_audio[0]
#         print (frames)
#         raw_samples = poses_audio[0]
#         with frames_deque_lock:
#             frames_deque.extend(frames)

#         #  frame.to_ndarray()
#         sound = pydub.AudioSegment(
#             data=raw_samples.tobytes(),
#             sample_width=frame.format.bytes,
#             frame_rate=frame.sample_rate,
#             channels=len(frame.layout.channels),
#         )
#         # Return empty frames to be silent.
#         new_frames = []
#         for frame in frames:
#             input_array = frame.to_ndarray()
#             new_frame = av.AudioFrame.from_ndarray(
#                 np.zeros(input_array.shape, dtype=input_array.dtype),
#                 layout=frame.layout.name,
#             )
#             new_frame.sample_rate = frame.sample_rate
#             new_frames.append(new_frame)
#         sound = poses_audio[0]
#         channel_sounds = sound.split_to_mono()
#         channel_samples = [s.get_array_of_samples() for s in channel_sounds]
#         new_samples: np.ndarray = np.array(channel_samples).T
#         new_samples = new_samples.reshape(raw_samples.shape)

#         new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
#         new_frame.sample_rate = frame.sample_rate
#         return new_frame

#         return new_frames

#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text-w-video",
#         mode=WebRtcMode.SENDRECV,
#         queued_audio_frames_callback=queued_audio_frames_callback,
#         video_frame_callback=video_frame_callback,
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
#         media_stream_constraints={"video": True, "audio": True},
#     )

#     status_indicator = st.empty()

#     if not webrtc_ctx.state.playing:
#         return

#     status_indicator.write("Loading...")
#     text_output = st.empty()
#     stream = None

#     while True:
#         if webrtc_ctx.state.playing:
#             if stream is None:
                

#                 stream = generated_flow[0]

#                 # status_indicator.write("Model loaded.")

#             sound_chunk = pydub.AudioSegment.empty()

#             audio_frames = []
#             with frames_deque_lock:
#                 while len(frames_deque) > 0:
#                     frame = frames_deque.popleft()
#                     audio_frames.append(frame)

#             if len(audio_frames) == 0:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue

#             status_indicator.write("Running. Say something!")

#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound

#             if len(sound_chunk) > 0:
#                 sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
#                     1#model.sampleRate()
#                 )
#                 buffer = np.array(sound_chunk.get_array_of_samples())
#                 # stream.feedAudioContent(buffer)
#                 # text = stream.intermediateDecode()
#                 text_output.markdown(f"**Text:** {text}")
#         else:
#             status_indicator.write("Stopped.")
#             break


# app_sst_with_video()


