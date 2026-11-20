import json
import os
import tempfile
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
        verbose="auto",
    )
    # Keras accepts integer verbosity at runtime; its installed annotations say str.
    validation_results = model.evaluate(
        validation_dataset,
        verbose=0,  # pyright: ignore[reportArgumentType]
        return_dict=True,
    )
    validation_accuracy = float(validation_results["accuracy"])
    predictions = model.predict(
        validation_dataset,
        verbose=0,  # pyright: ignore[reportArgumentType]
    )
    predicted_labels = np.argmax(predictions, axis=1)
    confusion = confusion_matrix(split, predicted_labels)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "misclassified_images.json").unlink(missing_ok=True)
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
    save_misclassification_report(
        split.validation_paths,
        split.validation_labels,
        predicted_labels,
        artifacts_dir / "misclassified_images.json",
    )
    print(f"検証データの正解率: {validation_accuracy:.1%}")
    print(f"モデルと学習結果を保存しました: {artifacts_dir}")


def save_misclassification_report(
    image_paths: tuple[Path, ...],
    true_labels: tuple[int, ...],
    predicted_labels: np.ndarray,
    output_path: Path,
) -> None:
    misclassified_images = [
        {
            "image_path": str(image_path.resolve()),
            "true_label": CLASS_LABELS[true_label],
            "predicted_label": CLASS_LABELS[int(predicted_label)],
        }
        for image_path, true_label, predicted_label in zip(
            image_paths, true_labels, predicted_labels, strict=True
        )
        if true_label != predicted_label
    ]
    report = {
        "validation_count": len(image_paths),
        "misclassified_images": misclassified_images,
    }
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized_report)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def confusion_matrix(split: ImageSplit, predicted_labels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(CLASS_DIRECTORIES), len(CLASS_DIRECTORIES)), dtype=np.int64)
    true_labels = np.asarray(split.validation_labels, dtype=np.int64)
    np.add.at(result, (true_labels, predicted_labels), 1)
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
    axes.imshow(matrix, cmap="Blues", origin="upper")
    axes.set(
        xticks=range(len(CLASS_LABELS)),
        yticks=range(len(CLASS_LABELS)),
        xticklabels=CLASS_DIRECTORIES,
        yticklabels=CLASS_DIRECTORIES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
    )
    axes.set_xticks(np.arange(1, matrix.shape[1]) - 0.5, minor=True)
    axes.set_yticks(np.arange(1, matrix.shape[0]) - 0.5, minor=True)
    axes.grid(which="minor", color="#64748b", linewidth=0.8)
    axes.tick_params(which="minor", length=0)
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
