import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time

st.set_page_config(page_title="Real-time Prediction", layout="wide")
st.subheader("Real-time Prediction")

#retrive the data
with st. spinner("Retriving Data from Redis database ..."):
    redis_face_db = face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)
st.success("Data retrieved successfully!")

#time
waitTime = 30 #seconds
setTime = time.time()
real_time_pred = face_rec.RealTimePrediction()

#Real time prediction
#streamlit webrtc

#callback function
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        global setTime
        
        img = frame.to_ndarray(format="bgr24")
        
        # You can manipulate the image here, e.g., flipping it.
        # flipped = img[::-1, :, :] 
        
        pred_img = real_time_pred.face_prediction(img, redis_face_db, 'Facial_Features', ['Name', 'Role'], thresh = 0.5)
        
        timenow = time.time()
        difftime = timenow - setTime
        if difftime >= waitTime:
            real_time_pred.save_log_redis()
            setTime = time.time()
            print('Log saved')
        
        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# Use the VideoProcessor class with webrtc_streamer
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)