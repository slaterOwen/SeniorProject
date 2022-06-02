# Import libraries #

import kerastuner as kt
from tensorboard.plugins.hparams import api as hp
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

%pip install keras-tuner - -upgrade

%load_ext tensorboard

# Download data #

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    'flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Create Dataset #

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

# Configure dataset #

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize data #

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Create model def and params for hypertuning #

num_classes = len(class_names)
!rm - rf ./logs/

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.3, 0.4))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )


def train_test_model(hparams):
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        layers.Dense(num_classes),
        layers.Dropout(hparams[HP_DROPOUT]),
    ])
    model.compile(optimizer=hparams[HP_OPTIMIZER],
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    epochs = 1
    history = model.fit(
        train_ds,
        epochs=epochs
    )
    _, accuracy = model.evaluate(val_ds)
    return accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

# Create/run grid search #


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1

# View results #

%tensorboard - -logdir logs/hparam_tuning

# Build the model used for random search #


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('num_units', [16, 32, 64]),
        activation='relu'))
    model.add(keras.layers.Dense(
        hp.Choice('dropout', [0.3, 0.4]),
        activation='relu'))
    model.add(keras.layers.Dense(
        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
        activation='relu'))

    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse', metrics=["accuracy"])
    return model

# Build the kerastuner for random search #


tuner = kt.RandomSearch(
    build_model,
    overwrite=True,
    objective=['accuracy'],
    max_trials=8)

# Train and run the tuner #

tuner.search(train_ds,
             validation_data=val_ds,
             epochs=1,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])


# Get and evaluate the results #

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

best_hyperparameters.get("num_units")

best_hyperparameters.get("dropout")

best_hyperparameters.get("learning_rate")
