# import streamlit as st
# import cv2
# st.markdown(
#     """
#     <style>
#     .text-box {
#         width: 700px;
#         font-size: 28px;
#         text-align: justify;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.header("Text Generation by Hand Gesture")

# cp = cv2.VideoCapture(0)
# stop = st.button("Stop")

# col1, col2 = st.columns([3,2])

# with col1:
#     st.markdown('<div class="text-box">It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair...</div>', unsafe_allow_html=True)

# frameplace = col2.empty()

# while True and not stop:
#     success, frame = cp.read()
#     if not success:
#         st.write("Frame not successful")
#         break
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frameplace.image(frame, width=700)

# cp.release()

import streamlit as st
import cv2

st.markdown(
    """
    <style>
    .text-box {
        font-size: 20px;
        width: 100%;
    }
    .frame {
        height: 400px;
        width: 100%;
    }
    .zoom {
        transform: scale(1.0);  
        transform-origin: 0 0; 
        width: 300%;             
        height: 300%;            
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="zoom">', unsafe_allow_html=True)

st.header("Text Generation by Hand Gesture")


cp = cv2.VideoCapture(0)
stop = st.button("Stop")
col1, col2 = st.columns([3, 2])

with col1:
    txt = st.text_input(
        "Text to analyze",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, (...)",
        key="text-box"
    )
    st.write(f"You wrote {len(txt)} characters.")

with col2:
    frameplace = st.empty()

while True and not stop:
    success, frame = cp.read()
    if not success:
        st.write("Frame not successful")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameplace.image(frame, use_column_width=True)  

cp.release()

st.markdown('</div>', unsafe_allow_html=True)
