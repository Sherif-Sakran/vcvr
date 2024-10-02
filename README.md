# Video-Conference-Voice-Recognition
# Speaker Identification

This project aims at identifying people in video conferences sreamed from a single device. Instead of showing the device's name, this project is done to display the speaker's name. To ahcieve this, this project utilizes both the Gaussian Mixure Model (GMM) and a Convolutional Neural Network (CNN).To utilize the speaker identification module, you can do follow the following steps:

# 1. Enrolling speakers
The enrollment process includes both creating the speakers' models of the audio recordings in the directory "speaker_identification/enrolment_recordings/" this is done by running the script enrol_speaker.py

# 2. Removing speakers (if needed)
This removes the GMM speaker model of the specified speaker, and it re-trains the neural network after removing this speaker (if train_CNN flag is True). This is done by running the script remove_speaker.py

# 3. Testing in real time
The real_time_backend_testing.py script recognizes the persons enrolled earlier using two different models: GMM and CNN.

# 4. Real-time testing using the extension
This part utilizes the GMM and CNN to recognize the previously enrolled speakers using our chrome extension. This is done through the real_time_testing.py
