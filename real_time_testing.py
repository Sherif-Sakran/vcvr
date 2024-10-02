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

import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

def extract_features(audio,rate): 
    coefs = 22
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,coefs,nfft = 1200, appendEnergy = True)
    mfcc_feature = preprocessing.scale(mfcc_feature)

    combined = np.hstack((mfcc_feature,)) 

    return combined


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

same_output_file = False
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3.1
record_len = RECORD_SECONDS
record = True
recognize = True
max_speaker_threshold = 0.4
condition = True
counter = 50
load_models = True
last_speaker = ""
weight = 3






while True:
    # program starts reading the paths from here
    path = "Add path here"

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
        print(path)
        print("GMM: ", max_speaker_name, " - ", max_speaker_probability)
        print("CNN: ", predicted_directory, " - ", predictions.max())
        time.sleep(1)
