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
<<<<<<< Updated upstream
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
=======
        _input = tf.keras.Input(shape=self.input_shape)
        x = layers.RandomFlip("horizontal")(_input)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        # ----Stage 0: Conv3x3, Stride 2, 24 channels------
        x = layers.Conv2D(24, (3, 3), activation="relu", padding="same", strides=2)(x)
        x = layers.BatchNormalization()(x)

        # ------Stage 1: Fused-MBConv1, 2 layers, Stride 1, 24 channels------
        for _ in range(2):
            x = self._Fused_MBConvBlock(
                x, out_channels=24, kernel_size=3, strides=1, use_se=False, r=4, t=1
            )
        # -------Stage 2: Fused-MBConv4, 4 layers, Stride 2, 48 channels------
        for i in range(4):
            stride = 2 if i == 0 else 1  # Chỉ giảm ảnh ở layer đầu tiên của stage
            x = self._Fused_MBConvBlock(x, out_channels=48, strides=stride, t=4)
        # -------Stage 3: Fused-MBConv4, 4 layers, Stride 2, 64 channels------
        for i in range(4):
            stride = 2 if i == 0 else 1
            x = self._Fused_MBConvBlock(x, out_channels=64, strides=stride, t=4)
        # -------Stage 4: MBConv4, 6 layers, Stride 2, 128 channels------
        for i in range(6):
            stride = 2 if i == 0 else 1
            x = self._MBConvBlock(x, out_channels=128, strides=stride, use_se=True, t=4)
        ## Stage 5: MBConv6, 9 layers, Stride 1, 160 channels, SE
        for _ in range(9):
            x = self._MBConvBlock(
                x, out_channels=160, kernel_size=3, strides=1, use_se=True, r=4, t=6
            )
        # Stage 6: MBConv6, 15 layers, Stride 2, 256 channels, SE
        for i in range(15):
            stride = 2 if i == 0 else 1
            x = self._MBConvBlock(x, out_channels=256, strides=stride, use_se=True, t=6)
        x = layers.Conv2D(1280, (1, 1), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)

        x = layers.GlobalMaxPooling2D()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=_input, outputs=outputs)
>>>>>>> Stashed changes
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
