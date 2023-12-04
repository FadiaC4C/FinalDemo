#@Fadia Ghamdi

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import sys
import time
import os

sys.path.append('Module-1')
from voice import *
sys.path.append('Module-2')
from OCR import *
sys.path.append('Module-3')
from classification import *
sys.path.append('Module-4')
from traffic_light import *
sys.path.append('../')

def set_page_config():
    st.set_page_config(
        page_title='Caption an Image',
        page_icon=':camera:',
        layout='wide',
    )

def initialize_model():
    try:
        hf_model = "Salesforce/blip-image-captioning-large"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = BlipProcessor.from_pretrained(hf_model)
        model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)
        return processor, model, device
    except Exception as e:
        st.error(f"Error initializing the model: {e}")
        return None, None, None

def generate_caption(processor, model, device, image):
    try:
        inputs = processor(image, return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None

def save_frame(frame, folder, prefix=""):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(folder, f"{prefix}_{timestamp}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) ####
        return image_path
    except Exception as e:
        st.error(f"Error saving frame: {e}")
        return None

def cleanup_temp_files(temp_files):
    try:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")

def main():
    # Add the picture to the header
    image = Image.open('header.png')
    st.image(image, use_column_width=True)

    # Load Camera
    cap = cv2.VideoCapture(1)  # Use camera index 0 for the default laptop camera and 1 from iphone camera

    # Change the IP address and port to match your DroidCam settings
    #droidcam_url = 'http://192.168.43.61:4747/video'
    #cap = cv2.VideoCapture(droidcam_url)

    # Use st.sidebar for left-aligned buttons
    with st.sidebar:
        # Create styled buttons
        button_caption = st.button(
            "Image Captioning",
            key="button_caption",
            help="Click to perform Image Captioning",
        )
        button_ocr = st.button(
            "OCR and Capture Image",
            key="button_ocr",
            help="Click to perform OCR and Capture Image",
        )
        button_traffic_sign = st.button(
            "Classify Traffic Sign",
            key="button_traffic_sign",
            help="Click to classify Traffic Sign",
        )
        button_traffic_light = st.button(
            "Classify Traffic Light",
            key="button_traffic_light",
            help="Click to classify Traffic Light",
        )
        button_exit = st.button(
            "Exit",
            key="button_exit",
            help="Click to exit the application",
        )

    mode = None  # Initialize mode

    # Delay before capturing the first frame
    time.sleep(1)

    video_placeholder = st.image([], channels="RGB", use_column_width=True)
    
    # Initialize the model outside the loop
    processor, model, device = initialize_model()

    temp_files = []  # Temporary files list for cleanup

    while True:
        ret, frame = cap.read()

        # Check if the frame is empty
        if not ret or frame is None:
            st.error("Error reading frame from the camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display camera feed in Streamlit
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        if button_ocr:
            if not ret:
                st.error("Failed to capture frame from the camera.")
            else:
                frame_path = save_frame(frame, ".", prefix="ocr_frame")
                if frame_path:
                    st.write("Frame saved:", frame_path)
                    ocr(frame_path)  
                    temp_files.append(frame_path)

        elif button_caption:
            if processor and model and device:
                caption = generate_caption(processor, model, device, frame)
                if caption:
                    playAudio(caption)

        elif button_traffic_sign:
            img_path = save_frame(frame, ".", prefix="captured_frame")
            if img_path:
                playAudio(classify(img_path))
                temp_files.append(img_path)

        elif button_traffic_light:
            img_path = save_frame(frame, ".", prefix="captured_frame")
            if img_path:
                traffic_lig(img_path)
                temp_files.append(img_path)

        elif button_exit:
            cleanup_temp_files(temp_files)
            break

        time.sleep(0.1)  # Add a small delay of 0.1 seconds

    cap.release()  # Release the camera

if __name__ == '__main__':
    main()




