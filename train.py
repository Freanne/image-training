import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import backend as K
from tensorflow import keras

# Logging setup
logging.basicConfig(level=logging.INFO)

def save_plot(figure, filename):
    """Save plot to disk."""
    path = os.path.join("plots", filename)
    os.makedirs("plots", exist_ok=True)
    figure.savefig(path)
    logging.info(f"Plot saved to {path}")

def plot_history(history, model_name):
    """Plot training history."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    
    # Plot loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    # Save the figure instead of showing
    save_plot(fig, f"{model_name}_history.png")

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, np.argmax(y_pred, axis=-1))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    # Save the confusion matrix plot
    save_plot(fig, f"{model_name}_confusion_matrix.png")

# Model Definitions

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
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Conv2D(16, kernel_size=5, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Conv2D(120, kernel_size=5, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(84, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

# Callback Definition
def create_callbacks(model_name):
    """Create Keras callbacks for training."""
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"{model_name}_best_model.weights.h5", save_best_only=True, monitor="val_loss"
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
    )
    return [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

# Train Model
def train_model(model, train_data, val_data, epochs, model_name):
    """Train the given model and return history."""
    callbacks = create_callbacks(model_name)
    logging.info(f"Training {model_name}...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
    )
    
    # Plot and save training history
    plot_history(history, model_name)
    
    return history

# Evaluate Model
def evaluate_model(model, test_data, model_name):
    """Evaluate the model and plot confusion matrix."""
    logging.info(f"Evaluating {model_name}...")
    
    # Get the ground truth labels and predictions
    y_true = test_data.labels
    y_pred = model.predict(test_data)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, model_name)

    return model.evaluate(test_data)

# Main Training Loop
def run_experiments(train_data, val_data, test_data, input_shape, num_classes, epochs):
    """Run multiple model experiments."""
    models_dict = {
        "EfficientNetB0": create_model_0(input_shape, num_classes),
        "EfficientNetB0_Pretrained": create_model_0_1(input_shape, num_classes),
        "CustomModel1": create_model_1(input_shape, num_classes),
        "InceptionLikeModel": create_model_2(input_shape, num_classes),
        "ResNetLikeModel": create_model_3(input_shape, num_classes),
        "LeNet5": create_model_4(input_shape, num_classes),
    }

    results = {}

    for model_name, model in models_dict.items():
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Train the model
        history = train_model(model, train_data, val_data, epochs, model_name)
        
        # Evaluate the model
        test_loss, test_accuracy = evaluate_model(model, test_data, model_name)
        
        results[model_name] = {"test_loss": test_loss, "test_accuracy": test_accuracy}
        
        logging.info(f"Model: {model_name} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    return results

if __name__ == "__main__":
    # Hyperparameters
    input_shape = (224, 224, 3)  # Taille des images
    num_classes = 4 # Nombre de classes à prédire (ex: 10 pour CIFAR-10)
    batch_size = 32
    epochs = 3  # Nombre d'époques pour l'entraînement

    # Chemins vers les datasets (ces dossiers doivent contenir les images classées par sous-dossiers)
    train_dir = "./splitted_folder/train"
    val_dir = "./splitted_folder/val"
    test_dir = "./splitted_folder/test"

    # Prétraitement et augmentation des images
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  # Normaliser les pixels entre 0 et 1
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Chargement des données avec augmentation
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Ne pas mélanger pour correspondre les prédictions et les étiquettes
    )

    # Exécution des expériences avec les modèles définis
    results = run_experiments(train_data, val_data, test_data, input_shape, num_classes, epochs)

    # Affichage des résultats
    for model_name, metrics in results.items():
        logging.info(f"{model_name} -> Loss: {metrics['test_loss']}, Accuracy: {metrics['test_accuracy']}")
