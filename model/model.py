import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os


class CNNModel:
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

    def build_model(self):
        model = models.Sequential(
            [
                layers.Input(shape=self.input_shape),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    def train(self, data, epochs=50, batch_size=16):
        # Assuming data is a dictionary with keys 'X_train', 'y_train', 'X_val', 'y_val'
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        )
        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler, early_stop],
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, img):
        """
        img: Một ảnh duy nhất đã được load bằng cv2 và resize về (128, 128)
        """
        img_batch = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img_batch, verbose=0)
        probability = prediction[0][0]
        class_id = 1 if probability > 0.5 else 0
        return class_id, probability

    def save(self, file_path="models/face_verify_v1.keras"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save(file_path)
        print(f"--- Đã lưu model thành công tại: {file_path} ---")
