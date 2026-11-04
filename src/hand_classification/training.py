import json
from pathlib import Path

import keras
import numpy as np

from hand_classification.data import (
    CLASS_DIRECTORIES,
    CLASS_LABELS,
    IMAGE_SIZE,
    ImageSplit,
    make_image_dataset,
    split_image_paths,
)
from hand_classification.model import create_model


def train_model(
    data_dir: Path,
    artifacts_dir: Path,
    epochs: int = 60,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 123,
) -> None:
    if epochs < 1:
        raise ValueError("epochs は1以上にしてください。")
    if batch_size < 1:
        raise ValueError("batch-size は1以上にしてください。")

    split = split_image_paths(data_dir, validation_split, seed)
    keras.utils.set_random_seed(seed)
    train_dataset = make_image_dataset(
        split.train_paths,
        split.train_labels,
        batch_size,
        seed=seed,
    )
    validation_dataset = make_image_dataset(
        split.validation_paths,
        split.validation_labels,
        batch_size,
    )

    model = create_model()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )
    validation_results = model.evaluate(validation_dataset, verbose=0, return_dict=True)
    validation_accuracy = float(validation_results["accuracy"])
    predictions = model.predict(validation_dataset, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    confusion = confusion_matrix(split, predicted_labels)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifacts_dir / "hand_classifier.keras")
    metadata = {
        "class_directories": list(CLASS_DIRECTORIES),
        "class_labels": list(CLASS_LABELS),
        "image_size": list(IMAGE_SIZE),
    }
    (artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_training_plot(history.history, artifacts_dir / "training_history.png")
    save_confusion_matrix(confusion, artifacts_dir / "confusion_matrix.png")
    print(f"検証データの正解率: {validation_accuracy:.1%}")
    print(f"モデルと学習結果を保存しました: {artifacts_dir}")


def confusion_matrix(split: ImageSplit, predicted_labels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(CLASS_DIRECTORIES), len(CLASS_DIRECTORIES)), dtype=np.int64)
    np.add.at(result, (split.validation_labels, predicted_labels), 1)
    return result


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(history.get("accuracy", []), label="Training")
    axes.plot(history.get("val_accuracy", []), label="Validation")
    axes.set(xlabel="Epoch", ylabel="Accuracy", title="Training history")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 6))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes, label="Image count")
    axes.set(
        xticks=range(len(CLASS_LABELS)),
        yticks=range(len(CLASS_LABELS)),
        xticklabels=CLASS_DIRECTORIES,
        yticklabels=CLASS_DIRECTORIES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
    )
    axes.invert_yaxis()
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
