import streamlit as st 
from tensorflow.keras.models import load_model
import cv2
import numpy as np 
import tempfile
from collections import Counter
from streamlit_webrtc import webrtc_streamer,VideoProcessorBase,WebRtcMode


ml=load_model("D:\\Waste management\\DATASET\\waste_management.h5")
class_name=["Non-Recyclabel","Organic","Recyclable"]

st.set_page_config(page_title="Waste Classification",layout="wide")

st.title(" Waste classification ")

def draw_predict(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,threh=cv2.threshold(blur,60,255,cv2.thresh_BINARY)
    conour,_=cv2.findcontours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in conour:
        area=cv2.contourArea(c)
        if area>1000:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame(x,y),(x+w,y+h),(255,0,0),3)
        imgg=cv2.resize(frame,224,224)
        imgg=imgg/255.0
        exp_dim=np.expand_dims(imgg,axis=0)
        prd=ml.predict(exp_dim)
        response=np.argmax(prd)
        label=class_name[response]
        cv2.putText(frame,f"{label}-{response}",(15,25),cv2.Font_HERSHEY_SIMPLEX,2,(0,255,0),2)
        return frame,label

tab1,tab2,tab3=st.tabs(["upload Video","webcam","chatbot"])

with tab1:
    st.header("Upload the video")
    upl=st.file_uploader("Upload the video ",type=["mp4","mkv"])
    prd=[]
    if upl:
        vid_fil=tempfile.NamedTemporaryfile(delete=False)
        vid_file.write(upl.read())
        cv2.videoCapture(vid_fil.name)
        sf=st.empty()
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            frame,label=draw_predict(frame)
            prd.append(label)

            frame=cv2.resize(frame,(640,320))
            sf.image(frame,channels="BGR")
        cap.release()
        if prd:
            fn=Counter(prd).most_common(1)[0][0]
            st.success("The predicted class is {fn}")

with tab2:
    st.header("Live video feed Prediction")
    if "cam_prd" not in st.session_state:
        st.session_state.cam_prd=[]
        class simpleprocess(VideoProcessorBase):
            def recv(self,frame):
                img=frame.to_ndarray(format="bgr 24")
                img,labels=draw_predict(img)
                st.session_state.cam_prd.append(labels)
                
                return img
        
        webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=simpleprocess,
            media_stream_constraints={"video":True,"audio":False},
            async_processing=False,
        ) 
        if(st.button("Final Outcome")):
                if st.session_state.cam_prd:
                    final=Counter(st.session_state.cam_prd).most_common(1)[0][0]
                    st.success(f"The predicted class {final}")