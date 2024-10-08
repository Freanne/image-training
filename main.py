import os
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Importing required libraries.")

import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil



logging.info("Creating artefacts directory if it doesn't exist.")
os.makedirs("artefacts", exist_ok=True)

image_size = (180, 180)
batch_size = 20
epochs = 100

logging.info("Loading dataset from directory.")
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "img",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'  # Ensure labels are one-hot encoded
)

logging.info("Setting up data augmentation layers.")
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomRotation(0.2),
    layers.RandomRotation(0.3),
    layers.RandomRotation(0.4),
    layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

logging.info("Applying data augmentation to training dataset.")
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

logging.info("Prefetching training and validation datasets.")
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

train_size = tf_data.experimental.cardinality(train_ds).numpy()
val_size = tf_data.experimental.cardinality(val_ds).numpy()

logging.info(f"Number of images in training dataset: {train_size * batch_size}")
logging.info(f"Number of images in validation dataset: {val_size * batch_size}")

def make_model(input_shape, num_classes):
    logging.info("Building the model.")
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

logging.info("Creating the model.")
model = make_model(input_shape=image_size + (3,), num_classes=4)
#keras.utils.plot_model(model, show_shapes=True)

logging.info("Compiling the model.")
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "artefacts/best_model.keras", save_best_only=True, monitor="val_acc", mode="max"
    ),
]

logging.info("Starting model training.")
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

logging.info("Plotting the learning curve.")
plt.figure(figsize=(12, 8))
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

plt.savefig('artefacts/learning_curve.png')
plt.close()

logging.info("Making predictions on the validation dataset.")
val_predictions = model.predict(val_ds)
val_pred_labels = tf.argmax(val_predictions, axis=1)

logging.info("Getting true labels from the validation dataset.")
val_true_labels = tf.concat([tf.argmax(y, axis=1) for x, y in val_ds], axis=0)

logging.info("Computing the confusion matrix.")
val_cm = confusion_matrix(val_true_labels, val_pred_labels)

logging.info("Plotting the confusion matrix.")
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=val_cm)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title('Confusion Matrix - Validation Set')
plt.savefig('artefacts/confusion_matrix.png')

plt.close()

logging.info("Creating a zip archive of the artefacts directory.")
shutil.make_archive('artefacts', 'zip', 'artefacts')
