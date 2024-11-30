# Standard library imports
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


random_seed = 123456

APK_METADATA_PATH = "utils/metadata.csv"

IMG_SIZE = 128

# Where to save all training and testing data after splitting
DATA_DIR = "./data"
NUMPY_FILES_DIR = "./utils/100k_download/npy"
# Define timeframe of relevant apps
YEAR_START = 2010
YEAR_END = 2022


def assemble_arrays():
    """
    Prepares the data for a Keras classifier by loading data from train.txt and test.txt.
    """
    X = np.load(os.path.join(DATA_DIR, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(DATA_DIR, "y.npy"), allow_pickle=True)

    # Combine arrays to shuffle them together
    combined = list(zip(X, y))
    np.random.shuffle(combined)

    # Unzip the shuffled data
    X, y = zip(*combined)
    X, y = np.array(X), np.array(y)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    np.save(os.path.join(DATA_DIR, "dexray_without_tesseract/X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "dexray_without_tesseract/X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "dexray_without_tesseract/y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "dexray_without_tesseract/y_test.npy"), y_test)

    return X_train, X_test, y_train, y_test


def main():
    ## The following defines custom fit and predict functions, main body starts below
    # Define a custom fit function so that we can change the number of epochs.
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=50, restore_best_weights=True
    )

    print("Assembling X, y and temp ...")
    X_train, X_test, y_train, y_test = assemble_arrays()

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

    print("Training the model ...")
    model.fit(
        X_train,
        y_train,
        shuffle=True,
        epochs=200,  # TODO: hardcoded for now because just wanna try
        callbacks=[es_callback],
        verbose=2,
    )

    model.save("../models/model-50k-without-tesseract")


if __name__ == "__main__":
    main()
