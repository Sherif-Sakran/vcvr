import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

def calculate_delta(array, coefs=20):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,coefs))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    coefs = 22
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,coefs,nfft = 1200, appendEnergy = True)
    # print("mfcc_feature: ",mfcc_feature.shape)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print("preprocessed mfcc_feature: ",mfcc_feature.shape)    
    delta = calculate_delta(mfcc_feature, coefs)
    delta_delta = calculate_delta(delta, coefs)
    # print("delta: ",delta.shape)
    # combined = np.hstack((mfcc_feature,delta, delta_delta)) 
    combined = np.hstack((mfcc_feature,)) 
    # combined = np.hstack((mfcc_feature,delta)) 
    # print("combined mffcs: ",combined.shape)
    return combined
