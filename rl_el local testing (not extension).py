import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


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


extensionNew_dataset = True

def get_latest_file(directory):
    # Get list of files in the directory
    files = os.listdir(directory)
    if not files:
        return None  # Directory is empty
    
    # Full paths of all files in the directory
    file_paths = [os.path.join(directory, filename) for filename in files]
    
    # Get the latest file based on modification time
    latest_file = max(file_paths, key=os.path.getmtime)
    
    return latest_file

model_path = get_latest_file("models_cnn")
print(model_path)

training_mfcc_directory = "training_mfcc"
class_names = sorted(os.listdir(training_mfcc_directory))  # Assuming train_data_dir is the path to the training directory
modelpath = "models/"

loaded_model = tf.keras.models.load_model(model_path)


source   = "2-wav_testing/"   

same_output_file = True


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3.1
record_len = RECORD_SECONDS
wav_path = "output.wav"
record = True
vad = True
normalize = True
recognize = True
max_speaker_threshold = 0.4
condition = True
counter = 50
frames = []
load_models = True
last_speaker = ""
weight = 3
while True:
    # condition = False
    counter += 1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Started recording...", end="")

    if record:
        if frames:
            frames = frames[len(frames)//3:]
            record_len = RECORD_SECONDS - 2
        # record_len = RECORD_SECONDS
        data_bytes = b''
        # frames = []
        for i in range(0, int(RATE / CHUNK * record_len)):
            data = stream.read(CHUNK)
            frames.append(data)
            data_bytes += data


        stream.stop_stream()
        stream.close()
        p.terminate()
        # print(type(data_bytes))
        # sys.exit()
        if same_output_file:
            ext = ""
        else: 
            ext = "./recordings_dump/" + str(counter)
        wf = wave.open(ext+wav_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("Done recording! - F")
        # print(len(frames))

    # if vad:
    #     pass

    if normalize:
        with wave.open(ext+wav_path, 'rb') as wav_file:
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
        normalized_audio.export(ext+wav_path.split(".")[0]+".wav", format='wav')
        path = ext+wav_path.split(".")[0]+".wav"


    if recognize:
        if load_models:
            #path where training speakers will be saved


            gmm_files = [os.path.join(modelpath,fname) for fname in 
                        os.listdir(modelpath) if fname.endswith('.gmm')]

            #Load the Gaussian Models
            models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
            # print("Loaded Models: ", len(models))
            speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
                        in gmm_files]
            load_models = False
        


        file_path = path
        sr,audio = read(file_path)
        vector = extract_features(audio, sr)
        
        
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
            # Apply softmax to the log likelihood values
        winner=np.argmax(log_likelihood)
        likelihood_values = np.exp(log_likelihood - np.max(log_likelihood)) 
        likelihood_values /= np.sum(likelihood_values)
        winner_score = log_likelihood[winner]
        predicted_speaker = speakers[winner]
        
        # Find the indices of the top three likelihood values
        top_indices = np.argsort(log_likelihood)[-3:][::-1]  # Indices of top three likelihoods in descending order
        top_likelihoods = np.exp(log_likelihood[top_indices] - np.max(log_likelihood))  # Compute likelihood values
        top_likelihoods /= np.sum(top_likelihoods)
        # selected_speakers.append(speakers[top_indices[0]])
        # selected_likelihoods.append(top_likelihoods[0])
        max_speaker_probability = top_likelihoods[0]
        max_speaker_name = speakers[top_indices[0]]
        # if top_likelihoods[0] < max_speaker_threshold:
        #     speaker = speakers[0]
        #     likelihood = top_likelihoods[0]
        #     print(f"1. {speaker}: {likelihood}")
        #     print("skipped")
        #     continue
        # Print the top three speakers and their likelihood values
        # for rank, idx in enumerate(top_indices):
        #     speaker = speakers[idx]
        #     likelihood = top_likelihoods[rank]
        #     print(f"{rank+1}. {speaker}: {likelihood}")
        # print(f"{predicted_speaker}")
        # print("Time taken in ms: ", sample_time*1000)

        # print(counter)
        # if max_speaker_probability >= 0.65:
        #     print("Speaker: ", max_speaker_name, " - ", max_speaker_probability)
        #     last_speaker = max_speaker_name
        #     if max_speaker_name == last_speaker:
        #         weight += 1
        # elif max_speaker_probability >= 0.55 or (max_speaker_name == last_speaker and max_speaker_probability >= 0.45 and weight > 3):
        #     print("1sec\t2sec\t3sec")
        #     print(selected_speakers)
        #     print(selected_likelihoods)
        #     last_speaker = max_speaker_name
        #     if max_speaker_name == last_speaker:
        #         weight += 1
        #     print("Max speaker: ", max_speaker_name, " - ", max_speaker_probability)
        # else:
        #     print("Last speaker: ", last_speaker)
        #     print("Current speaker: ", max_speaker_name, " - ", max_speaker_probability)
        #     weight = 0
            # last_speaker = selected_speakers[0]
        # print("GMM: ", max_speaker_name, " - ", max_speaker_probability)
            
        # Test on a new example (assuming new_example is a 2D vector)
        new_example = vector
        new_example = new_example[:299, :]
        # Ensure new_example has the shape (1, 299, 22) to match the input shape expected by the model
        new_example = np.expand_dims(new_example, axis=0)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(new_example)

        # Convert predictions to class labels
        predicted_class = np.argmax(predictions)
        # print(ext.split('/')[-1])
        # Now, you can use the predicted class label to get the corresponding directory name
        class_to_directory = {i: class_names[i] for i in range(len(class_names))}
        predicted_directory = class_to_directory[predicted_class]
        # if predictions.max() > 0.98 or  (predictions.max() > 0.95 and weight > 3):
        #     print(predicted_directory, " - ", predictions.max())
        #     if predicted_directory == last_speaker:
        #         weight += 1
        #     last_speaker = predicted_directory
        # elif predictions.max() > 0.90 or (predictions.max() >= 0.75 and predicted_directory == last_speaker and weight > 3):
        #     if last_speaker == predicted_directory:
        #         weight += 1
        #     print("Max speaker: ", predicted_directory," - ", predictions.max())
        # else:
        #     print("Last speaker: ", last_speaker," - ", predictions.max())
        #     weight = 0

        print("GMM: ", max_speaker_name, " - ", max_speaker_probability)
        print("CNN: ", predicted_directory, " - ", predictions.max())
    # time.sleep(1)
