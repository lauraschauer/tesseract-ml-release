# Standard library imports
import argparse
import csv
import datetime
import os
import random as python_random
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm

# Optional import (commented)
# import tensorflow_addons as tfa

# Local imports
from tesseract import evaluation, metrics, mock, temporal, spatial


random_seed = 123456

APK_METADATA_PATH = "/scratch/users/mbenali/metadata.csv"

GOODWARE_PATH = "/scratch/users/mbenali/download_apk/100k_download/images/goodware"
MALWARE_PATH = "/scratch/users/mbenali/download_apk/100k_download/images/malware"

IMG_SIZE=128

NUMPY_FILES_DIR = "/scratch/users/mbenali/download_apk/100k_download/npy"
# Define timeframe of relevant apps 
YEAR_START = 2010
YEAR_END = 2022


def load_apk_metadata(file_path):
    """Loads the APK metadata from the CSV file into a dictionary for quick lookups."""
    metadata = {}
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=[
            'sha256', 'sha1', 'md5', 'date_time', 'number1',
            'package', 'number2', 'number3', 'dex_date', 'number4', 'source'
        ])
        csv_file.seek(0)  # Reset file pointer to the beginning
        for row in reader:
            metadata[row['sha256']] = row['dex_date']
    return metadata

def get_date_time_from_hash(search_hash, metadata):
    """
    Retrieves the `dex_date` for the given hash using preloaded metadata."""
    if search_hash in metadata:
        try:
            return datetime.strptime(metadata[search_hash], '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            print(f"Error parsing date for {search_hash}: {e}")
            return None
    return None


def assemble_arrays(metadata):
    """Assembles numpy arrays from .npy files along with their labels and dates."""
    X = []
    y = []
    temp = []

    dirs = [NUMPY_FILES_DIR + '/malware', NUMPY_FILES_DIR + '/goodware']
    stop = 0

    for directory in dirs:
        if not os.path.exists(directory):
            continue

        # List files and wrap with tqdm for a progress bar
        files = [file for file in os.listdir(directory) if file.endswith('.npy')]
        for file in tqdm(files, desc=f"Processing {directory}"): 
            if stop >= 100:
                break
            stop += 1

            if not file.endswith('.npy'):
                continue

            # Obtain the date
            apk_date = get_date_time_from_hash(file[:-4], metadata)  # remove .npy

            # Do not include if outside of date range
            if apk_date is None or apk_date.year < YEAR_START or apk_date.year > YEAR_END:
                continue

            temp.append(apk_date)

            filepath = os.path.join(directory, file)
            array = np.load(filepath)
            X.append(array.flatten())  # Flattening ensures all arrays are rows

            # Obtain the label 
            y.append(0 if 'goodware' in directory else 1)

        print(f"Done with {directory}")

    return np.stack(X), np.array(y), np.array(temp)


def main():
    ## The following defines custom fit and predict functions, main body starts below
    # Define a custom fit function so that we can change the number of epochs.
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=50, restore_best_weights=True
    )

    def fit_with_epochs(X_train, y_train):
        model.fit(
            X_train,
            y_train,
            shuffle=True,
            epochs=200, # TODO: hardcoded for now because just wanna try
            callbacks=[es_callback],
            verbose=2,
        )

    def predict_keras(X_test):
        probabilities = model.predict(X_test, verbose=0)
        return (probabilities > 0.5).astype(int).flatten()  # Convert to 1D array of labels

    ## Start of actual main
    print("Getting metadata to assemble X, y and temp ...")
    metadata = load_apk_metadata(APK_METADATA_PATH)
    print("Assembling X, y and temp ...")
    X, y, temp = assemble_arrays(metadata)

    # Replicate the DexRay model
    model_architecture = Sequential()
    model_architecture.add(
        Conv1D(
            filters=64,
            kernel_size=12,
            activation="relu",
            input_shape=(IMG_SIZE * IMG_SIZE, 1),
        )
    )
    model_architecture.add(MaxPooling1D(pool_size=12))
    model_architecture.add(Conv1D(filters=128, kernel_size=12, activation="relu"))
    model_architecture.add(MaxPooling1D(pool_size=12))
    model_architecture.add(Flatten())
    model_architecture.add(Dense(64, activation="sigmoid"))
    model_architecture.add(Dense(1, activation="sigmoid"))

    model = keras.models.clone_model(model_architecture)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
    )

    # Get Tesseract test and train split 
    print("Getting Tesseract time aware splits ...")
    splits = temporal.time_aware_train_test_split(
        X, y, temp, train_size=12, test_size=2, granularity='month'
    )

    print("Training the model ...")
    results = evaluation.fit_predict_update(
        model, 
        *splits,
        fit_function=fit_with_epochs,
        predict_function=predict_keras,
    )
    print(results)

if __name__ == "__main__":
    main()
