import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from config import config
import prediction_pipeline_runner

st.header("Text Generator with Gestures", divider='rainbow')   

col1, col2 = st.columns([2, 2])
with col1:
    st.header("Generated Text")
    st.text_area(config.SESSION_TEXT)

with col2:
    st.header("Generated Image")
    st.image(config.SESSION_IMG)