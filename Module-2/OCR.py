
######============================================================================================



import sys

import streamlit as st
import numpy as np
from PIL import Image
import time
import utlis
import easyocr
import cv2
import os
from gtts import gTTS
import pygame.mixer
from utils import evaluate_word
from hmm import HMM

def get_image(img):
    image = Image.open(img)
    return np.asarray(image)

def get_reader_lang(lang='ar'):
    reader = easyocr.Reader([lang])
    return reader

def get_result(image, lang='ar'):
    reader = get_reader_lang(lang)
    result = reader.readtext(image)
    return result 

def play_audio(caption):
    model = HMM()
    arabic_caption = model.diacritized_word(caption)

    audio_file = "caption_arabic.mp3"
    # Create the gTTS object and save the audio file:
    tts = gTTS(text=arabic_caption, lang='ar')
    tts.save(audio_file)

    pygame.mixer.init()
    sound = pygame.mixer.Sound(audio_file)
    sound.play()

@st.cache(allow_output_mutation=True)
def cached_get_result(image, lang='ar'):
    return get_result(image, lang)

def main():
    st.markdown("<h1 style='text-align: center;position:static; top:-50 ;color: rgb(65, 59, 59);font-size:300%;'>Arabic OCR</h1>",
                unsafe_allow_html=True)
    side_bar = st.sidebar
    image_file = side_bar.file_uploader('',type=['png', 'jpg', 'jpeg'])
    col1, col2 = st.columns(2)
    
    if image_file is not None:
        image= get_image(image_file)
        start_btn = side_bar.button('Start OCR')
        
        if start_btn:
            animation_placeholder = st.empty()
            animation_placeholder.markdown(utlis.read_custum_html('animation_style.html'), unsafe_allow_html=True)
            t1 = time.time()
            result = cached_get_result(image)
            animation_placeholder.empty()
            annoted_image = utlis.annotate_image(image, result)
            annoted_image = Image.fromarray(annoted_image)
            col1.text('Image')
            col1.image( annoted_image, use_column_width=True)
            extracted_text = utlis.get_raw_text(result)
            col2.text('Extracted text')
            col2.write(extracted_text,  use_column_width=True)
            t2 = time.time()
            print(f'The task done in {t2-t1} sec')
            play_audio(extracted_text)

if __name__ == "__main__":
    main()
