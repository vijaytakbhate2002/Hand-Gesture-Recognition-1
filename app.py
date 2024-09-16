from processes.video_frame import getResultFromModel
import cv2
import streamlit as st

st.header("Hand Gesture Recognition", divider='rainbow')
st.sidebar.header("Model Tuning (Playground)")
confirmation_ratio = st.sidebar.slider(label="Select Model Confirmation Rate", min_value=0.1, max_value=1.0)
confirmation_ratio = 0.7
speed = st.sidebar.slider(label="Prediction Speed", min_value=3, max_value=97, value=80)
speed_to_methodical_scale = (99 - speed)/100
methodical = st.sidebar.slider(label="Methodical", min_value=0.0, max_value=1.0, value=speed_to_methodical_scale, disabled=True)
confirmation_list_len = 100 - speed

col1, col2 = st.columns(2)
with col1:
    text_placeholder = st.empty()
with col2:
    image_placeholder = st.empty()

out = getResultFromModel(confirmation_ratio=confirmation_ratio, confirmation_list_len=confirmation_list_len, camera_num=1)

while True:
    result = next(out)
    frame, text, prediction_result, success, char = (val for val in result.values())
    if success:
        text_placeholder.title(text)
    image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

