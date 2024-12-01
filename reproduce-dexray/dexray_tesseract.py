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


IMG_SIZE=128
# Where to save all training and testing data after splitting 
DATA_DIR = "./data/dexray_tesseract_10_percent_malware"


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
    print("Assembling X, y and temp ...")
    X = np.load("data/X.npy", allow_pickle=True)
    y = np.load("data/y.npy", allow_pickle=True)
    temp = np.load("data/temp.npy", allow_pickle=True)

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
        X, y, temp, train_size=12, test_size=1, granularity='month'
    )

    X_train, X_test, y_train, y_test, t_train, t_test = splits

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(DATA_DIR, 't_train.npy'), t_train)
    np.save(os.path.join(DATA_DIR, 't_test.npy'), t_test)
    print(f"Arrays saved to {DATA_DIR}")

    print("Training the model ...")
    results = evaluation.fit_predict_update(
        model, 
        *splits,
        fit_function=fit_with_epochs,
        predict_function=predict_keras,
    )
    print(results)

    model.save("../models/model-50k-tesseract-10-percent-malware")

if __name__ == "__main__":
    main()
