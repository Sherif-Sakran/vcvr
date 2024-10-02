import pyaudio
import wave
import sounddevice
import sys
from tqdm import tqdm
from IPython.display import Audio
from pydub import AudioSegment
import os

speaker_name = "Silence"

recording_length = 55

file_path = os.path.join("enrolment_recordings", speaker_name +".wav")
# file_path = "" + speaker_name +".wav"
output_file = file_path.split('.')[0]+ '.wav'
output_folder = output_file.split('.')[0]

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = int(recording_length) + .1
WAVE_OUTPUT_FILENAME = speaker_name + ".wav"



record = True
vad = False

if record:
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Started recording...")
    frames = []

    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)


    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Done recording! - T")