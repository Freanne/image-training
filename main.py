import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Importing required libraries.")
import keras
from keras import layers
from tensorflow import data as tf_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
# from sklearn.utils import class_weight
from tensorflow.keras.applications import EfficientNetB0

import itertools
import numpy as np
import matplotlib.pyplot as plt




logging.info("Creating artefacts directory if it doesn't exist.")
os.makedirs("artefacts", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
image_size = ( 260 , 260 )
batch_size = 64
epochs = 5

logging.info("Loading dataset from directory.")
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.25,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'
)

logging.info("Setting up data augmentation layers.")
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1)
]

def data_augmentation(images):
    for layer in data_augmentation_layers :
        images = layer(images)
    return images

logging.info("Applying data augmentation to training dataset.")
def apply_data_augmentation(img, label):
    return data_augmentation(img), label

#logging.info("Applying data augmentation to training dataset.")
train_ds = train_ds.map(apply_data_augmentation, num_parallel_calls=tf_data.AUTOTUNE)


logging.info("Prefetching training and validation datasets.")
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

train_size = tf_data.experimental.cardinality(train_ds).numpy()
val_size = tf_data.experimental.cardinality(val_ds).numpy()

logging.info(f"Number of images in training dataset: {train_size * batch_size}")
logging.info(f"Number of images in validation dataset: {val_size * batch_size}")

logging.info("Counting the number of images per class in the training dataset.")
class_counts = {}
for _, labels in train_ds.unbatch():
    class_idx = labels.numpy()
    if class_idx in class_counts:
        class_counts[class_idx] += 1
    else:
        class_counts[class_idx] = 1

logging.info(f"Number of images per class in the training dataset: {class_counts}")

class_labels = [f"Class {i}" for i in range(len(class_counts))]
class_names = train_ds.class_names
print(class_names)
logging.info(f"Class names: {class_names}")

total_images = sum(class_counts.values())
class_weights = {class_idx: total_images / (len(class_counts) * count) for class_idx, count in class_counts.items()}


def create_model_0(input_shape, num_classes):
    logging.info("Building Model 0 - EfficientNetB0.")
    model = EfficientNetB0(
        include_top=True,
        weights=None,
        classes=num_classes,
        input_shape=input_shape
    )
    return model
    
def create_model_0_1(input_shape, num_classes):
    logging.info("Building Model 0_1 - EfficientNetB0 with pretrained weights.")
    inputs = layers.Input(shape=input_shape)  # Input layer
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    logging.info("Freezing the pretrained weights.")
    model.trainable = False

    # Rebuild top
    logging.info("Rebuilding the top layers.")
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    logging.info("Compiling the model.")
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def create_model_1(input_shape, num_classes):
    logging.info("Building Model 1.")
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

    

def create_model_2(input_shape, num_classes):
    logging.info("Building Model 2 - Inception-Like Model.")
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)

    # First block
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Inception-like block
    tower_1 = layers.Conv2D(64, 1, padding="same", activation="relu")(x)
    tower_1 = layers.Conv2D(128, 3, padding="same", activation="relu")(tower_1)
    
    tower_2 = layers.Conv2D(64, 1, padding="same", activation="relu")(x)
    tower_2 = layers.Conv2D(128, 5, padding="same", activation="relu")(tower_2)
    
    tower_3 = layers.MaxPooling2D(3, strides=1, padding="same")(x)
    tower_3 = layers.Conv2D(128, 1, padding="same", activation="relu")(tower_3)

    x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    x = layers.BatchNormalization()(x)

    # Additional blocks
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def create_model_3(input_shape, num_classes):
    logging.info("Building Model 3 - ResNet-Like Model.")
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
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        residual = layers.BatchNormalization()(residual)  # Ensure batch normalization is applied to the residual

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
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


def create_model_4(input_shape, num_classes):
    logging.info("Building Model 4 - LeNet-5.")
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(6, kernel_size=5, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(16, kernel_size=5, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def create_model_5(input_shape, num_classes):
    logging.info("Building Model 5 - VGG-16 (light).")
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def create_model_6(input_shape, num_classes):
    logging.info("Building Model 6 - MobileNetV2 (light).")
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def plot_confusion_matrix(cm, class_names, title='Confusion matrix', cmap=plt.cm.Blues, file_name='confusion_matrix.png'):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({100 * cm[i, j] / cm[i, :].sum():.2f}%)',
                 horizontalalignment="center", verticalalignment='bottom',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)
    
    
models = [
    create_model_0, 
    create_model_0_1, 
    create_model_1, create_model_2, 
    create_model_3, create_model_4, create_model_5, 
    create_model_6
    ]

for i, model_fn in enumerate(models):
    logging.info(f"Creating Model {i+1}.")
    model = model_fn(input_shape=image_size + (3,), num_classes=4)

    logging.info(f"Compiling Model {i+1}.")
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f"models/best_model_{i+1}.keras", save_best_only=True, monitor="val_acc", mode="max"
        ),
        keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True),
        keras.callbacks.CSVLogger(f'artefacts/training_log_{i+1}.csv')  # Save training log
    ]

    logging.info(f"Starting Model {i+1} training.")
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
        class_weight=class_weights
    )

    logging.info(f"Plotting the learning curve for Model {i+1}.")
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve - Model {i+1}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'artefacts/learning_curve_{i+1}.png')
    plt.savefig(f'graphs/learning_curve_{i+1}.png')
    plt.close()

    logging.info(f"Making predictions on the validation dataset for Model {i+1}.")
    val_predictions = model.predict(val_ds)
    val_pred_labels = tf.argmax(val_predictions, axis=1)

    logging.info(f"Getting true labels from the validation dataset for Model {i+1}.")
    val_true_labels = tf.concat([y for _, y in val_ds], axis=0)

    logging.info(f"Computing the confusion matrix for Model {i+1}.")
    val_cm = confusion_matrix(val_true_labels, val_pred_labels)

    logging.info(f"Plotting the confusion matrix for Model {i+1}.")
    plot_confusion_matrix(val_cm, class_names, f'Confusion Matrix - Validation Set - Model {i+1}', cmap=plt.cm.Blues, file_name=f'artefacts/confusion_matrix_{i+1}.png')
    plot_confusion_matrix(val_cm, class_names, f'Confusion Matrix - Validation Set - Model {i+1}', cmap=plt.cm.Blues, file_name=f'graphs/confusion_matrix_{i+1}.png')

    logging.info(f"Saving model weights for Model {i+1}.")
    model.save_weights(f'models/model_weights_{i+1}.weights.h5')
    #model.save_weights(f'models/model_{i+1}.weights.h5')

logging.info("Creating a zip archive of the artefacts directory.")
shutil.make_archive('artefacts', 'zip', 'artefacts')
shutil.make_archive('models', 'zip', 'models')
shutil.make_archive('graphs', 'zip', 'graphs')
