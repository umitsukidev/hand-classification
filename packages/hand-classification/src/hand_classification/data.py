from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_DIRECTORIES = ("rock", "scissors", "paper")
CLASS_LABELS = ("グー", "チョキ", "パー")
IMAGE_SIZE = (96, 96)
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ImageSplit:
    train_paths: tuple[Path, ...]
    train_labels: tuple[int, ...]
    validation_paths: tuple[Path, ...]
    validation_labels: tuple[int, ...]


def split_image_paths(
    data_dir: Path,
    validation_split: float,
    seed: int,
) -> ImageSplit:
    if not 0 < validation_split < 1:
        raise ValueError(
            "validation-split は 0 より大きく 1 より小さい値にしてください。"
        )
    if not data_dir.is_dir():
        raise ValueError(f"データディレクトリが見つかりません: {data_dir}")

    actual_directories = {path.name for path in data_dir.iterdir() if path.is_dir()}
    expected_directories = set(CLASS_DIRECTORIES)
    missing = expected_directories - actual_directories
    extra = actual_directories - expected_directories
    if missing or extra:
        details = []
        if missing:
            details.append(f"不足: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"未対応: {', '.join(sorted(extra))}")
        expected = ", ".join(CLASS_DIRECTORIES)
        raise ValueError(
            f"クラスディレクトリは {expected} にしてください（{'、'.join(details)}）。"
        )

    random_generator = np.random.default_rng(seed)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    validation_paths: list[Path] = []
    validation_labels: list[int] = []

    for label, directory_name in enumerate(CLASS_DIRECTORIES):
        class_dir = data_dir / directory_name
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if len(image_paths) < 2:
            raise ValueError(
                f"{class_dir} には画像が2枚以上必要です（現在: {len(image_paths)}枚）。"
            )

        shuffled_indices = random_generator.permutation(len(image_paths))
        validation_count = round(len(image_paths) * validation_split)
        validation_count = max(1, min(len(image_paths) - 1, validation_count))
        validation_indices = shuffled_indices[:validation_count]
        train_indices = shuffled_indices[validation_count:]

        train_paths.extend(image_paths[index] for index in train_indices)
        train_labels.extend([label] * len(train_indices))
        validation_paths.extend(image_paths[index] for index in validation_indices)
        validation_labels.extend([label] * len(validation_indices))

    return ImageSplit(
        train_paths=tuple(train_paths),
        train_labels=tuple(train_labels),
        validation_paths=tuple(validation_paths),
        validation_labels=tuple(validation_labels),
    )


def make_image_dataset(
    image_paths: tuple[Path, ...],
    labels: tuple[int, ...],
    batch_size: int,
    seed: int | None = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in image_paths], list(labels))
    )
    if seed is not None:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_example(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = load_image_tensor(path, IMAGE_SIZE)
        return image, label

    return (
        dataset.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_image_tensor(
    image_path: tf.Tensor | str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tf.Tensor:
    if isinstance(image_path, Path):
        image_path = str(image_path)
    path = tf.convert_to_tensor(image_path, dtype=tf.string)
    image_data = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False,
    )
    image.set_shape((None, None, 3))
    return tf.image.resize(image, image_size)
