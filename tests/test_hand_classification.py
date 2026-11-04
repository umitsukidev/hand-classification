import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from hand_classification.cli import predict_image
from hand_classification.data import CLASS_DIRECTORIES, CLASS_LABELS, split_image_paths
from hand_classification.training import train_model


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((12, 12, 3), color, dtype=np.uint8)
    tf.io.write_file(str(path), tf.io.encode_png(pixels))


def create_dataset(root: Path, images_per_class: int = 2) -> None:
    colors = ((220, 20, 20), (20, 220, 20), (20, 20, 220))
    for directory, color in zip(CLASS_DIRECTORIES, colors, strict=True):
        for index in range(images_per_class):
            write_image(root / directory / f"sample-{index}.png", color)


class DataSplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_contains_every_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            create_dataset(data_dir, images_per_class=4)

            first = split_image_paths(data_dir, validation_split=0.25, seed=123)
            second = split_image_paths(data_dir, validation_split=0.25, seed=123)

            self.assertEqual(first, second)
            self.assertEqual(set(first.train_labels), {0, 1, 2})
            self.assertEqual(set(first.validation_labels), {0, 1, 2})
            self.assertFalse(set(first.train_paths) & set(first.validation_paths))

    def test_missing_classes_and_too_few_images_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "rock").mkdir()
            with self.assertRaisesRegex(ValueError, "クラスディレクトリ"):
                split_image_paths(data_dir, validation_split=0.2, seed=123)

            create_dataset(data_dir, images_per_class=1)
            with self.assertRaisesRegex(ValueError, "2枚以上"):
                split_image_paths(data_dir, validation_split=0.2, seed=123)


class TrainingSmokeTests(unittest.TestCase):
    def test_train_save_reload_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "hand_data"
            artifacts_dir = root / "artifacts"
            create_dataset(data_dir)

            train_model(
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                epochs=1,
                batch_size=3,
                validation_split=0.5,
                seed=123,
            )

            for artifact_name in (
                "hand_classifier.keras",
                "metadata.json",
                "training_history.png",
                "confusion_matrix.png",
            ):
                self.assertTrue((artifacts_dir / artifact_name).is_file())

            label, probabilities = predict_image(
                data_dir / "rock" / "sample-0.png",
                artifacts_dir,
            )
            self.assertIn(label, CLASS_LABELS)
            self.assertEqual(len(probabilities), len(CLASS_LABELS))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
