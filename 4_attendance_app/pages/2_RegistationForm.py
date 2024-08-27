import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import face_rec

st.set_page_config(page_title="Registration Form", layout="wide")
st.subheader("Registration Form")

#registration form
registration_form = face_rec.RegistrationForm()

#collect person name and role
person_name = st.text_input(label = 'Name', placeholder = 'Enter your name')
role = st.selectbox(label='Role', options = ['Student', 'Teacher'])

#collect facial embeding 
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")
    reg_img, embedding = registration_form.get_embedding(img)
    #save data in local txt
    if embedding is not None:
        with open('face_embedding.txt', 'ab') as f:
            np.savetxt(f, embedding)
    
    
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24") 

webrtc_streamer(key="registration", video_frame_callback=video_callback_func)

#save data in redis database


if st.button('Submit'):
    return_Value = registration_form.save_data_redis(person_name, role)
    if return_Value == 'No data to save':
        st.error('No data to save')
    elif return_Value == 'Name cannot be empty':
        st.error('Name cannot be empty')
    else:
        st.success('Data saved successfully!')
        