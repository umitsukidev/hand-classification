import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypedDict

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


class PlotAxesBounds(TypedDict):
    left: float
    top: float
    right: float
    bottom: float


class ConfusionMatrixPlot(TypedDict):
    image_width: int
    image_height: int
    axes_bounds: PlotAxesBounds


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

    class_labels = split.class_directories
    model = create_model(class_count=len(split.class_directories))
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
        "class_directories": list(split.class_directories),
        "class_labels": list(class_labels),
        "image_size": list(IMAGE_SIZE),
    }
    (artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_training_plot(history.history, artifacts_dir / "training_history.png")
    confusion_matrix_plot = save_confusion_matrix(
        confusion,
        artifacts_dir / "confusion_matrix.png",
        class_labels=class_labels,
    )
    save_misclassification_report(
        split.validation_paths,
        split.validation_labels,
        predicted_labels,
        artifacts_dir / "misclassified_images.json",
        confusion_matrix_plot=confusion_matrix_plot,
        class_labels=class_labels,
    )
    print(f"検証データの正解率: {validation_accuracy:.1%}")
    print(f"モデルと学習結果を保存しました: {artifacts_dir}")


def save_misclassification_report(
    image_paths: tuple[Path, ...],
    true_labels: tuple[int, ...],
    predicted_labels: np.ndarray,
    output_path: Path,
    confusion_matrix_plot: ConfusionMatrixPlot | None = None,
    class_labels: tuple[str, ...] = CLASS_LABELS,
) -> None:
    misclassified_images = []
    correctly_classified_images = []
    for image_path, true_label, predicted_label in zip(
        image_paths, true_labels, predicted_labels, strict=True
    ):
        image_record = {
            "image_path": str(image_path.resolve()),
            "true_label": class_labels[true_label],
            "predicted_label": class_labels[int(predicted_label)],
        }
        if true_label == predicted_label:
            correctly_classified_images.append(image_record)
        else:
            misclassified_images.append(image_record)

    report: dict[str, object] = {
        "validation_count": len(image_paths),
        "misclassified_images": misclassified_images,
        "correctly_classified_images": correctly_classified_images,
    }
    if confusion_matrix_plot is not None:
        report["confusion_matrix_plot"] = confusion_matrix_plot
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
    class_count = len(split.class_directories)
    if predicted_labels.ndim != 1 or len(predicted_labels) != len(
        split.validation_labels
    ):
        raise ValueError("予測件数が検証データ件数と一致しません。")
    if np.any(predicted_labels < 0) or np.any(predicted_labels >= class_count):
        raise ValueError("モデルの出力クラス数が学習データと一致しません。")
    result = np.zeros((class_count, class_count), dtype=np.int64)
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


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    class_labels: tuple[str, ...] = CLASS_DIRECTORIES,
) -> ConfusionMatrixPlot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("混同行列は正方行列である必要があります。")
    if matrix.shape[0] != len(class_labels):
        raise ValueError("混同行列とクラスラベルの数が一致しません。")

    configure_japanese_font(plt)
    class_count = len(class_labels)
    figure, axes = plt.subplots(
        figsize=(max(7, class_count * 1.2), max(6, class_count))
    )
    axes.imshow(matrix, cmap="Blues", origin="upper")
    axes.set(
        xticks=range(class_count),
        yticks=range(class_count),
        xticklabels=class_labels,
        yticklabels=class_labels,
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
    figure.canvas.draw()
    image_width, image_height = figure.canvas.get_width_height()
    axes_bounds = axes.get_position()
    plot_geometry: ConfusionMatrixPlot = {
        "image_width": image_width,
        "image_height": image_height,
        "axes_bounds": {
            "left": float(axes_bounds.x0),
            "top": float(1.0 - axes_bounds.y1),
            "right": float(axes_bounds.x1),
            "bottom": float(1.0 - axes_bounds.y0),
        },
    }
    figure.savefig(
        output_path,
        dpi=figure.dpi,
        bbox_inches=figure.bbox_inches,
    )
    plt.close(figure)
    return plot_geometry


def configure_japanese_font(plt: Any) -> None:
    """Select an already installed Japanese font when one is available."""
    from matplotlib import font_manager

    preferred_names = (
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Yu Gothic",
        "Hiragino Sans",
        "IPAexGothic",
        "IPAGothic",
    )
    installed_names = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_names:
        if font_name in installed_names:
            plt.rcParams["font.family"] = font_name
            break
