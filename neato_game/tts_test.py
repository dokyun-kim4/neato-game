from gtts import gTTS
from tempfile import NamedTemporaryFile
from playsound import playsound

player = 324
mytext = f'{player}번 탈락'
  
language = 'ko'
  
tts = gTTS(text=mytext, lang=language) 

f = NamedTemporaryFile()
tts.write_to_fp(f)

playsound(f.name, block=False)