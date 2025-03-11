from pydub import AudioSegment
from pydub.playback import play

def play_alert():
    sound = AudioSegment.from_file("alert.mp3")
    play(sound)
