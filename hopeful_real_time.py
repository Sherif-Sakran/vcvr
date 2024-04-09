import pyaudio
import wave
import subprocess
import sounddevice
import sys
import time
import os

from pydub import AudioSegment


import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
import sys

source   = "2-wav_testing/"   

#path where training speakers will be saved
modelpath = "4-gmm_models/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
print("Loaded Models: ", len(models))
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]



CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3.1

wav_path = "output.wav"
record = False
vad = True
normalize = False
recognize = True

condition = True
while condition:
    condition = False
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Started recording...")

    if record:
        frames = []
        data_bytes = b''
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            data_bytes += data


        stream.stop_stream()
        stream.close()
        p.terminate()
        print(type(data_bytes))
        # sys.exit()
        
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("Done recording! - F")

    if vad:
        pass

    if normalize:
        with wave.open(wav_path, 'rb') as wav_file:
            # Get the audio file properties
            sample_width = wav_file.getsampwidth()
            num_channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            # Read the audio data
            audio_data = wav_file.readframes(num_frames)


        # Convert the audio data to AudioSegment object
        audio = AudioSegment(
            data=audio_data,
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=num_channels
        )

        # Normalize the volume
        normalized_audio = audio.normalize()

        # Change the sampling rate to 1
        normalized_audio = normalized_audio.set_frame_rate(16000)

        # Combine all channels to one channel
        normalized_audio = normalized_audio.set_channels(1)

        # Export the normalized audio as WAV file
        normalized_audio.export(wav_path.split(".")[0]+"_normalized.wav", format='wav')
        path = wav_path.split(".")[0]+"_normalized.wav"


    if recognize:
        # path = wav_path.split(".")[0]+"_normalized.wav"
        path = wav_path
        print("Testing Audio : ", path)
        time1 = time.time()   
        sr,audio = read(path)
        vector   = extract_features(audio,sr)
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner=np.argmax(log_likelihood)
        winner_score = log_likelihood[winner]
        predicted_speaker = speakers[winner]
        sample_time = time.time() - time1
        print(f"Predicted speaker: {predicted_speaker}")
        print("Time taken in ms: ", sample_time*1000)
    time.sleep(1)
