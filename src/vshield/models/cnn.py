"""Compact binary CNN for anti-spoof experiments."""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from vshield.models.augmentation import build_training_augmentation


class CNNModel:
    def __init__(self, input_shape=(128, 128, 3), seed=42):
        self.input_shape = input_shape
        self.seed = seed
        self.model = self.build_model()
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    def build_model(self):
        return models.Sequential(
            [
                layers.Input(shape=self.input_shape),
                build_training_augmentation(self.seed),
                layers.Conv2D(32, 3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Conv2D(128, 3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def train(self, data, epochs=25, batch_size=16):
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
            ),
        ]
        return self.model.fit(
            data["X_train"],
            data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

    def evaluate(self, images, labels):
        return self.model.evaluate(images, labels, return_dict=True)

    def predict(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        probability = float(prediction[0][0])
        return (1 if probability > 0.5 else 0), probability

    def save(self, file_path="artifacts/models/face_verify_v2.keras"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save(file_path)
        print(f"Saved model: {file_path}")
