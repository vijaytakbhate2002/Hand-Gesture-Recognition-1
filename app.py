from processes.video_frame import getResultFromModel
import cv2
import streamlit as st

st.header("Hand Gesture Recognition", divider='rainbow')
st.sidebar.header("Model Tuning (Playground)")
confirmation_rate = st.sidebar.slider(label="Select Model Confirmation Rate", min_value=0.0, max_value=1.0)

out = getResultFromModel(camera_num=0)
col1, col2 = st.columns(2)
with col1:
    text_placeholder = st.empty()
with col2:
    image_placeholder = st.empty()

while True:
    result = next(out)
    frame, text, prediction_result, success, char = (val for val in result.values())
    cv2.imshow("Frame", frame)
    if success:
         text_placeholder.title(text)
    image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break


