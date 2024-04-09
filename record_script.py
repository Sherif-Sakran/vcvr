import pyaudio
import wave
import sounddevice
import sys
from tqdm import tqdm

print("script starts here")
speaker_name = input("Name: ")
# two recordings in Arabic and two in English. each recording is 60 seconds
recording_number = input("Recording id (e1, e2, a1, a2): ")
# recording_length = int(input("Length in seocnds: "))
recording_length = 60




CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = int(recording_length) + 5.1
WAVE_OUTPUT_FILENAME = speaker_name + "_" + recording_number +".wav"

file_path = "dataset_cm/" + WAVE_OUTPUT_FILENAME

record = True


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