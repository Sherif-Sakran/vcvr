load_vad_model = True
import pyaudio
import wave
import sounddevice
import sys
from tqdm import tqdm
from IPython.display import Audio
from pydub import AudioSegment
import os


record = False
vad = False
segment = True
normalize = False
duplicate_for_mfcc = True
generate_speaker_model = True
generate_mfcc = True
train_cnn = True # only if the next step is testing not another enrollment
num_epochs = 30
batch_size = 32


recording_length = 240

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = int(recording_length) + .1


enrolment_directory = "enrolment_recordings"
enrolment_paths = os.listdir(enrolment_directory)
enrolment_paths

training_directory = "training"
if not os.path.exists(training_directory):
    os.makedirs(training_directory)

training_mfcc_directory = "training_mfcc"
if not os.path.exists(training_mfcc_directory):
    os.makedirs(training_mfcc_directory)


for enrolment_recording in enrolment_paths:
    file_path = os.path.join(enrolment_directory, enrolment_recording)
    print(file_path)
    output_file = enrolment_recording.split('.')[0]+ '.wav'
    speaker_name = output_file.split('.')[0]
    output_folder = os.path.join(training_directory, speaker_name)
    print(output_folder)
    WAVE_OUTPUT_FILENAME = file_path
    if os.path.exists(output_folder):
        print("speaker previously enrolled")
        continue
  

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



    if vad:
        import torch
        if load_vad_model:
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                            model='silero_vad',
                                            force_reload=True,
                                            onnx=False)

            (get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks) = utils
            load_vad_model = False

        file = WAVE_OUTPUT_FILENAME
        Audio(file)

        # file_path = file_path.split('.')[0]+'_only_speech'+ '.wav'
        print(file)
        wav = read_audio(file, sampling_rate=RATE)
        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=RATE)
        # merge all speech chunks to one audio
        if speech_timestamps:
                save_audio(file_path,
                        collect_chunks(speech_timestamps, wav), sampling_rate=RATE) 
                Audio(file_path)
        else:
                print("No activity detected")


    if segment:
        from scipy.io import wavfile
        import os

        def trim_wav(input_file, output_folder, segment_name, offset=0):
            # Read the WAV file
            sample_rate, audio_data = wavfile.read(input_file)

            # Define the duration of each segment in samples (3 seconds)
            segment_duration = 3 * sample_rate

            # Calculate the number of segments
            num_segments = len(audio_data) // segment_duration
            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Trim the audio into segments
            for i in range(num_segments):
                start_sample = i * segment_duration
                end_sample = (i + 1) * segment_duration
                segment = audio_data[start_sample:end_sample]
                # Save each segment as a separate WAV file
                output_file = os.path.join(output_folder, f"{segment_name}_{i+1+offset}.wav")
                wavfile.write(output_file, sample_rate, segment)

            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Directory '{output_folder}' created successfully.")
        else:
            print(f"Directory '{output_folder}' already exists.")

        trim_wav(file_path, output_folder, output_folder.split('\\')[-1], 0)

    if normalize:
        import glob

        # Replace 'directory_path' with the actual directory path you want to read files from
        directory_path = output_folder

        # Use glob to get a list of all WAV files in the directory recursively
        wav_files = glob.glob(directory_path + '/**/*.wav', recursive=True)

        # Print the list of WAV files
        print(wav_files)
        for file_name in wav_files:
            with wave.open(file_name, 'rb') as wav_file:
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
            normalized_audio.export(file_name, format='wav')


    if duplicate_for_mfcc:
        import shutil

        def duplicate_directory(source_dir, destination_dir):
            try:
                shutil.copytree(source_dir, destination_dir)
                print(f"Directory '{source_dir}' duplicated to '{destination_dir}' successfully.")
            except FileExistsError:
                print(f"Error: Destination directory '{destination_dir}' already exists.")

        # Example usage:
        source_directory = output_folder
        destination_directory = output_folder+"_copy"

        duplicate_directory(source_directory, destination_directory)


    if generate_speaker_model:
        from scipy.io.wavfile import read
        from sklearn.mixture import GaussianMixture as GMM
        import numpy as np
        from FeatureExtraction import extract_features
        import pickle as cPickle

        directory = source_directory
        subset = ""
        dest = "models/"
        if not os.path.exists(dest):
            os.makedirs(dest)

        audios = subset
        file_paths = os.listdir(output_folder)
        file_paths = sorted(file_paths)
        # print(file_paths)
        # Extracting features for each speaker
        features = np.asarray(())
        for path in file_paths:
            path = directory + "/" + path    
            path = path.strip()   
            print (path)
            
            # read the audio
            sr,audio = read(audios+path)
            
            # extract 40 dimensional MFCC & delta MFCC features
            vector = extract_features(audio,sr)
            
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            print(features.shape)
            # when features of 5 files of speaker are concatenated, then do model training
            # -> if count == 5: --> edited below

            # speaker_dir = mfcc_dir + directory
            # speaker_name = path.split("/")[1].split(".")[0]

            # if not os.path.exists(speaker_dir):
            #     os.makedirs(speaker_dir)
            # # Assuming 'features' is the 2D array containing the features
            # csv_path = speaker_dir + "/" + speaker_name + ".csv"
            # np.savetxt(csv_path, vector, delimiter=",")
            # print("Features saved to", csv_path)
            

        gmm = GMM(n_components = 5, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        # dumping the trained gaussian model
        picklefile = directory.split('\\')[-1] + ".gmm"
        print(picklefile)
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
        features = np.asarray(())

    if generate_mfcc:
        import os
        import numpy as np
        from FeatureExtraction import extract_features
        from scipy.io.wavfile import read

        directory = output_folder
        wav_files = []

        # Traverse through all directories and subdirectories
        for root, dirs, files in os.walk(directory):
            # Check each file in the current directory
            for file in files:
                # Check if the file has a .wav extension
                if file.endswith('.wav'):
                    # If it does, add its full path to the list
                    wav_files.append(os.path.join(root, file))

            for filename in wav_files:
                print(os.path.join(root, filename))
                if filename.endswith(".wav"):
                    file_path = filename
                    sr,audio = read(file_path)
                    vector = extract_features(audio, sr)
                    csv_filename = os.path.splitext(filename)[0] + ".csv"
                    csv_path = csv_filename
                    np.savetxt(csv_path, vector, delimiter=",")
                    # os.remove(file_path)
                    print(f"Processed {filename} and saved the features to {csv_filename}")

        for file_path in wav_files:
            os.remove(file_path)


        def move_directory(source_dir, destination_dir):
            try:
                shutil.move(source_dir, destination_dir)
                print(f"Directory '{source_dir}' moved to '{destination_dir}' successfully.")
            except Exception as e:
                print(f"Error: {e}")

        # Example usage:
        source_directory = output_folder
        destination_directory = "training_mfcc/" +output_folder.split('\\')[-1]

        move_directory(source_directory, destination_directory)

        os.rename(source_directory+'_copy', source_directory.split('_copy')[0])


if train_cnn:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    import os
    import pandas as pd
    import numpy as np

    def load_data(directory, num_frames=299):
        data = []
        labels = []
        classes = os.listdir(directory)

        for i, class_name in enumerate(classes):
            class_dir = os.path.join(directory, class_name)
            files = os.listdir(class_dir)
            for file in files:
                file_path = os.path.join(class_dir, file)
                df = pd.read_csv(file_path, header=None)  # Assuming no header in CSV files
                # print(file_path)
                if df.shape[0] != 299:
                    continue
                # Reshape data to have (num_frames, num_features) shape
                reshaped_data = df.values.reshape((num_frames, -1))  
                data.append(reshaped_data)
                labels.append(i)  # Assigning numerical label to each class
            
        return np.array(data), np.array(labels)

    train_data_dir = training_mfcc_directory

    train_data, train_labels = load_data(train_data_dir)

    num_speakers = len(os.listdir(train_data_dir))

    # Confirm the shape of the data
    print("Train data shape:", train_data.shape)
    def create_cnn_model(input_shape, num_classes):
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(256, kernel_size=3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    num_speakers = len(os.listdir(train_data_dir))
    num_speakers
    # Define the CNN model
    model = create_cnn_model(input_shape=(299, 22), num_classes=num_speakers)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    # Train the model using the validation data for validation
    history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, 
                        validation_data=(val_data, val_labels))

    
    model_cnn_directory = "models_cnn"
    if not os.path.exists(training_mfcc_directory):
        os.makedirs(training_mfcc_directory)

    model_path = os.path.join(model_cnn_directory, f"SI_cnn_{num_speakers}_{num_epochs}_epochs_Demo.h5")
    model.save(model_path)
