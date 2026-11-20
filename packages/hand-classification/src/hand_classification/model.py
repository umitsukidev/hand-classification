import keras

from hand_classification.data import CLASS_DIRECTORIES, IMAGE_SIZE


def create_model(class_count: int = len(CLASS_DIRECTORIES)) -> keras.Model:
    if class_count < 2:
        raise ValueError("クラス数は2以上にしてください。")
    inputs = keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    images = keras.layers.RandomFlip("horizontal")(inputs)
    images = keras.layers.Rescaling(1.0 / 255)(images)
    images = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(images)
    images = keras.layers.MaxPooling2D()(images)
    images = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(images)
    images = keras.layers.MaxPooling2D()(images)
    images = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(images)
    images = keras.layers.MaxPooling2D()(images)
    images = keras.layers.Flatten()(images)
    images = keras.layers.Dense(64, activation="relu")(images)
    outputs = keras.layers.Dense(class_count, activation="softmax")(images)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
