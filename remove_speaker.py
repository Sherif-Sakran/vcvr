import os
import shutil

speaker_to_remove = "Youssef"

train_cnn = True # only if the next step is testing not another enrollment
num_epochs = 30
batch_size = 32

training_directory = "training"
training_mfcc_directory = "training_mfcc"

output_folder = os.path.join(training_directory, speaker_to_remove)

models_gmm = "models"

os.remove(os.path.join(models_gmm, speaker_to_remove+'.gmm'))
shutil.rmtree(os.path.join(training_directory, speaker_to_remove))
shutil.rmtree(os.path.join(training_mfcc_directory, speaker_to_remove))



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
