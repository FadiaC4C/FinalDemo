from gtts import gTTS
import os
import time
import pygame.mixer
from googletrans import Translator
from utils import evaluate_word
from hmm import HMM



def playAudio(caption):
    # Translate the caption to Arabic
    translator = Translator()


    translation = translator.translate(caption, dest='ar')
    
    # Add Tashkeel to the Arabic caption
    #tashkeela = Tashkeela()
    #arabic_caption = tashkeela.tashkeel(translation.text)
    model = HMM()
    arabic_caption = model.diacritized_word(translation.text)

    audio_file = "caption_arabic.mp3"
    # Create the gTTS object and save the audio file:
    tts = gTTS(text=arabic_caption, lang='ar')
    tts.save(audio_file)

    pygame.mixer.init()
    sound = pygame.mixer.Sound(audio_file)
    sound.play()



    time.sleep(5)  # sleep time as needed

    # Example 
#caption2 = 'I saw I girl  '
#play_audio(caption2)
#print(caption2)
