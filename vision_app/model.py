import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)
