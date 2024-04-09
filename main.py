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
import glob

notNormalized = False


f2f = 0
f2m = 0
m2f = 0
m2m = 0


test = "normalized"
subset = "testing_same"

# test = "normalized"
# subset = "testing_different"

source = f"../../dataset/{test}_dataset/{subset}/"

#path where training speakers will be saved
modelpath = "4-gmm_models"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
print("Loaded Models: ", len(models))
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

error = 0
total_samples = 0.0

external_class_true_count = 0
# print("Press '1' for checking a single Audio or Press '0' for testing a complete set of audio with Accuracy?")
# take=int(input().strip())
real_time = False
if real_time:
    path = sys.argv[1]
    print("Testing Audio : ", path)
    time1 = time.time()   
    total_samples+= 1.0
    
    # print("Testing Audio : ", path)
    sr,audio = read(source + path)
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
    # print("OK score: ", winner_score)

    # time.sleep(1.0)

    # time.sleep(1.0)
else:
    preprocessingMisPredictions = 0
    print("Testing the model with all the samples...")
    false_predictions = []
    test_file = "Testing_audio_Path.txt"       
    # file_paths = open(test_file,'r')
    file_paths = glob.glob(f"{source}/**/*.wav", recursive=True)

    # Read the test directory and get the list of test audio files 
    avg_time = 0
    total_time = 0
    for path in file_paths:
        time1 = time.time()   
        total_samples+= 1.0
        path=path.strip()
        # print("Testing Audio : ", path)
        sr,audio = read(path)
        vector   = extract_features(audio,sr)
        print(vector.shape)
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner=np.argmax(log_likelihood)
        winner_score = log_likelihood[winner]
        predicted_speaker = speakers[winner]
        speaker_label = path.split("/")[-2]
        # print("OK score: ", winner_score)
        if predicted_speaker != speaker_label:
            if speaker_label == "A" or speaker_label == "Y" or speaker_label == "S" or speaker_label == "O":
                if predicted_speaker == "A" or predicted_speaker == "Y" or predicted_speaker == "S" or predicted_speaker == "O":
                    m2m += 1
                else:
                    m2f += 1
            else:
                if predicted_speaker == "L" or predicted_speaker == "Renad" or predicted_speaker == "Reem":
                    f2f += 1
                else:
                    f2m += 1
            if speaker_label == "S":
                preprocessingMisPredictions += 1
            print(f"{speaker_label}:{predicted_speaker}")
            print(f"False Score: {winner_score}")
            print(path)
            error += 1
        # time.sleep(1.0)
        sample_time = time.time() - time1
        total_time += sample_time
    print (error, total_samples)
    accuracy = ((total_samples - error) / total_samples)

    print ("Accuracy: ", round(accuracy, 4))
    print(f"Average time taken per sample in ms", round((total_time/total_samples)*1000, 2))
    print("Preprocessing MisPredictions: ", preprocessingMisPredictions)
    print("f2f: ", f2f)
    print("f2m: ", f2m)
    print("m2f: ", m2f)
    print("m2m: ", m2m)
    # print ("The following Predictions were False :")
    # print (false_predictions)
    print(modelpath)
print ("Speaker Identified Successfully")
