from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import cv2  # OpenCV for image processing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU, Dropout
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


images_path = '/content/drive/MyDrive/CUBS2/IMAGES/'
lima_profiles_path = '/content/drive/MyDrive/CUBS2/SEGMENTATIONS/Manual-A1/'

def load_image(image_name):
    image_path = os.path.join(images_path, image_name)
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_coordinates(file_path):
    return np.loadtxt(file_path)

def coordinates_to_mask(coords, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    coords = coords.astype(np.int32)
    cv2.polylines(mask, [coords], isClosed=True, color=1, thickness=1)
    cv2.fillPoly(mask, [coords], color=1)
    return mask

def create_tube_mask(lumen_mask, media_mask):
    combined_mask = lumen_mask | media_mask
    contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hull = cv2.convexHull(contours[0])
        tube_mask = np.zeros_like(combined_mask)
        cv2.drawContours(tube_mask, [hull], -1, (1), thickness=cv2.FILLED)
        return tube_mask
    return combined_mask

def connect_ends_and_create_mask(lumen_coords, media_coords, image_shape):

    lumen_mask = np.zeros(image_shape, dtype=np.uint8)
    media_mask = np.zeros(image_shape, dtype=np.uint8)

    lumen_coords = lumen_coords.astype(np.int32)
    media_coords = media_coords.astype(np.int32)

    cv2.polylines(lumen_mask, [lumen_coords], isClosed=False, color=1, thickness=1)
    cv2.polylines(media_mask, [media_coords], isClosed=False, color=1, thickness=1)

    contour_coords = np.vstack((lumen_coords, media_coords[::-1]))  # Reverse the order of media to ensure they join correctly

    tube_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(tube_mask, [contour_coords], color=1)

    return tube_mask

def process_single_image(image_name):
    if not image_name.endswith('.tiff'):
        return None
    image = load_image(image_name)
    base_name = os.path.splitext(image_name)[0]
    lumen_file = os.path.join(lima_profiles_path, f"{base_name}-LI.txt")
    media_file = os.path.join(lima_profiles_path, f"{base_name}-MA.txt")
    lumen_coords = load_coordinates(lumen_file)
    media_coords = load_coordinates(media_file)
    tube_mask = connect_ends_and_create_mask(lumen_coords, media_coords, image.shape)
    return (image, tube_mask)

def load_data():
    data = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(process_single_image, os.listdir(images_path))
    for result in results:
        if result is not None:
            data.append(result)
    return data

data = load_data()

def preprocess_data(data, target_size=(256, 256)):
    X = []
    y = []
    for image, tube_mask in data:
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
        tube_resized = cv2.resize(tube_mask, target_size, interpolation=cv2.INTER_NEAREST)

        X.append(image_resized)
        y.append(tube_resized)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.uint8)

    X /= 255.0

    unique_values = np.unique(y)
    print(f"Unique values in masks: {unique_values}")

    y = y[..., np.newaxis]

    return X, y

def load_data_with_filenames():
    data = []
    filenames = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        file_list = list(os.listdir(images_path))  # converts map object to list
        results = executor.map(process_single_image, file_list)

    for idx, result in enumerate(results):
        if result is not None:
            data.append(result)
            filenames.append(file_list[idx])

    return data, filenames

data, filenames = load_data_with_filenames()

def save_filenames_to_file(filenames, file_path):
    with open(file_path, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")

X, y = preprocess_data(data)

X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(X, y, range(len(X)), test_size=0.2, random_state=42)

val_filenames = [filenames[i] for i in val_indices]
save_filenames_to_file(val_filenames, "validation_filenames.txt")

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

X, y = preprocess_data(data)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]


def combined_dice_bce_loss(y_true, y_pred, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
    """
    Combined Dice and Binary Cross-Entropy Loss.

    Parameters:
    - y_true: Ground truth labels (binary).
    - y_pred: Predicted labels.
    - dice_weight: Weight for Dice loss component.
    - bce_weight: Weight for Binary Cross-Entropy loss component.
    - smooth: Smoothing factor to avoid division by zero.

    Returns:
    - Combined loss value.
    """
    # Flattening the arrays
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Dice coefficient calculation
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    # Binary Cross-Entropy loss calculation
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Combine the two losses
    combined_loss = dice_weight * dice_loss + bce_weight * bce_loss
    return combined_loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def build_model(hp):
    inputs = Input((256, 256, 1))

    def convolution_operation(x, filters):
        x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = Dropout(rate=hp.Float('dropout_rate', 0.0, 0.5, step=0.1))(x)
        x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def encoder(x, filters):
        x_skip = convolution_operation(x, filters)
        x_encoded = MaxPooling2D(pool_size=(2, 2))(x_skip)
        return x_skip, x_encoded

    def decoder(x, x_skip, filters):
        x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate()([x, x_skip])
        x = convolution_operation(x, filters)
        return x

    filters = hp.Int('initial_filters', min_value=32, max_value=128, step=32)

    skip1, encoder_1 = encoder(inputs, filters)
    skip2, encoder_2 = encoder(encoder_1, filters*2)
    skip3, encoder_3 = encoder(encoder_2, filters*4)
    skip4, encoder_4 = encoder(encoder_3, filters*8)

    conv_bottle = convolution_operation(encoder_4, filters*16)

    decoder_1 = decoder(conv_bottle, skip4, filters*8)
    decoder_2 = decoder(decoder_1, skip3, filters*4)
    decoder_3 = decoder(decoder_2, skip2, filters*2)
    decoder_4 = decoder(decoder_3, skip1, filters)

    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(decoder_4)

    model = Model(inputs, outputs)

    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='unet_tuning',
    project_name='unet_opt'
)

# Assuming X_train, y_train, X_val, y_val are available
tuner.search(X_train, y_train, epochs=25, validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]

model.compile(optimizer='adam', loss=combined_dice_bce_loss, metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(patience=10, verbose=1, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1, monitor='val_loss'),
    ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1, monitor='val_loss')
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks
)
model.save("bestunet_dice.keras")