import streamlit as st
import cv2

st.header("Text Generation by hand gesture")
cp = cv2.VideoCapture(0)
stop = st.button("stop")
st.write("Press q to quite the video")
st.frameplace = st.empty()

while True and not stop:
    sucess, frame = cp.read()
    if not sucess:
        st.write("Frame not successed")
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.frameplace.image(frame)

cp.release()


