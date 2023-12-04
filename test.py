import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import playsound
from googletrans import Translator
import cv2

def set_page_config():
    st.set_page_config(
        page_title='Caption an Image', 
        page_icon=':camera:', 
        layout='wide',
    )

def initialize_model():
    hf_model = "Salesforce/blip-image-captioning-large"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = BlipProcessor.from_pretrained(hf_model)
    model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)
    return processor, model, device

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

def resize_image(image, max_width):
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        height = int(height * ratio)
        image = cv2.resize(image, (max_width, height))
    return image

def generate_caption(processor, model, device, image):
    inputs = processor(image, return_tensors='pt').to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def caption_to_audio(caption):
    # Translate the caption to Arabic
    translator = Translator()
    translation = translator.translate(caption, dest='ar')
    arabic_caption = translation.text

    # Generate Arabic audio
    tts = gTTS(text=arabic_caption, lang='ar')
    tts.save("caption_arabic.mp3")
    audio_file = "caption_arabic.mp3"
    return audio_file

def play_audio(audio):
    playsound.playsound(audio)

def main():
    set_page_config()
    st.header("Caption an Image :camera:")

    capture_button = st.button("Capture Image from Webcam")

    if capture_button:
        image = capture_image()
        image = resize_image(image, max_width=300)

        st.image(image, caption='Your image')

        with st.sidebar:
            st.divider() 
            if st.sidebar.button('Generate Caption'):
                with st.spinner('Generating caption...'):
                    processor, model, device = initialize_model()
                    caption = generate_caption(processor, model, device, image)
                    st.header("Caption:")
                    st.markdown(f'**{caption}**')

                    audio = caption_to_audio(caption)
                    play_audio(audio)

if __name__ == '__main__':
    main()
