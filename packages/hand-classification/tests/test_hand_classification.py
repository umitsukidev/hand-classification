import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tensorflow as tf

from hand_classification.cli import load_classifier, main, predict_image, serve_loop
from hand_classification.data import (
    CLASS_DIRECTORIES,
    CLASS_LABELS,
    make_image_dataset,
    split_image_paths,
)
from hand_classification.training import save_confusion_matrix, train_model


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


class ConfusionMatrixGeometryTests(unittest.TestCase):
    def test_plot_metadata_matches_full_png_axes_and_cell_centers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "confusion_matrix.png"
            geometry = save_confusion_matrix(
                np.asarray([[4, 0, 1], [2, 3, 0], [0, 1, 5]]),
                output_path,
            )

            with output_path.open("rb") as image_file:
                png_header = image_file.read(24)
            self.assertEqual(png_header[:8], b"\x89PNG\r\n\x1a\n")
            image_width, image_height = struct.unpack(">II", png_header[16:24])
            self.assertEqual(geometry["image_width"], image_width)
            self.assertEqual(geometry["image_height"], image_height)

            axes = geometry["axes_bounds"]
            self.assertGreater(axes["left"], 0.0)
            self.assertLess(axes["left"], axes["right"])
            self.assertLess(axes["right"], 1.0)
            self.assertGreater(axes["top"], 0.0)
            self.assertLess(axes["top"], axes["bottom"])
            self.assertLess(axes["bottom"], 1.0)

            axes_width = (axes["right"] - axes["left"]) * image_width
            axes_height = (axes["bottom"] - axes["top"]) * image_height
            self.assertAlmostEqual(axes_width, axes_height, places=5)

            first_cell_center = (
                axes["left"] + (axes["right"] - axes["left"]) / 6,
                axes["top"] + (axes["bottom"] - axes["top"]) / 6,
            )
            last_cell_center = (
                axes["left"] + (axes["right"] - axes["left"]) * 5 / 6,
                axes["top"] + (axes["bottom"] - axes["top"]) * 5 / 6,
            )
            self.assertLess(first_cell_center[0], last_cell_center[0])
            self.assertLess(first_cell_center[1], last_cell_center[1])


class TrainingSmokeTests(unittest.TestCase):
    def test_train_save_reload_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "hand_data"
            artifacts_dir = root / "artifacts"
            create_dataset(data_dir)
            validation_split = 0.5
            seed = 123
            split = split_image_paths(data_dir, validation_split, seed)

            train_model(
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                epochs=1,
                batch_size=3,
                validation_split=validation_split,
                seed=seed,
            )

            for artifact_name in (
                "hand_classifier.keras",
                "metadata.json",
                "training_history.png",
                "confusion_matrix.png",
                "misclassified_images.json",
            ):
                self.assertTrue((artifacts_dir / artifact_name).is_file())

            saved_model = load_classifier(artifacts_dir)
            validation_predictions = np.argmax(
                saved_model.predict(
                    make_image_dataset(
                        split.validation_paths,
                        split.validation_labels,
                        batch_size=3,
                    ),
                    verbose=0,  # pyright: ignore[reportArgumentType]
                ),
                axis=1,
            )
            expected_misclassifications = [
                {
                    "image_path": str(image_path.resolve()),
                    "true_label": CLASS_LABELS[true_label],
                    "predicted_label": CLASS_LABELS[int(predicted_label)],
                }
                for image_path, true_label, predicted_label in zip(
                    split.validation_paths,
                    split.validation_labels,
                    validation_predictions,
                    strict=True,
                )
                if true_label != predicted_label
            ]
            expected_correctly_classified = [
                {
                    "image_path": str(image_path.resolve()),
                    "true_label": CLASS_LABELS[true_label],
                    "predicted_label": CLASS_LABELS[int(predicted_label)],
                }
                for image_path, true_label, predicted_label in zip(
                    split.validation_paths,
                    split.validation_labels,
                    validation_predictions,
                    strict=True,
                )
                if true_label == predicted_label
            ]
            misclassification_report = json.loads(
                (artifacts_dir / "misclassified_images.json").read_text(
                    encoding="utf-8"
                )
            )
            confusion_matrix_plot = misclassification_report["confusion_matrix_plot"]
            self.assertGreater(confusion_matrix_plot["image_width"], 0)
            self.assertGreater(confusion_matrix_plot["image_height"], 0)
            self.assertEqual(
                set(confusion_matrix_plot["axes_bounds"]),
                {"left", "top", "right", "bottom"},
            )
            self.assertEqual(
                misclassification_report["validation_count"],
                len(split.validation_paths),
            )
            self.assertEqual(
                misclassification_report["misclassified_images"],
                expected_misclassifications,
            )
            self.assertEqual(
                misclassification_report["correctly_classified_images"],
                expected_correctly_classified,
            )
            self.assertEqual(
                len(misclassification_report["misclassified_images"])
                + len(misclassification_report["correctly_classified_images"]),
                misclassification_report["validation_count"],
            )

            label, probabilities = predict_image(
                data_dir / "rock" / "sample-0.png",
                artifacts_dir,
            )
            self.assertIn(label, CLASS_LABELS)
            self.assertEqual(len(probabilities), len(CLASS_LABELS))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)

            json_stdout = io.StringIO()
            with redirect_stdout(json_stdout):
                main(
                    [
                        "predict",
                        str(data_dir / "rock" / "sample-0.png"),
                        "--artifacts-dir",
                        str(artifacts_dir),
                        "--json",
                    ]
                )
            cli_payload = json.loads(json_stdout.getvalue())
            self.assertTrue(cli_payload["ok"])
            self.assertIn(cli_payload["predicted_label"], CLASS_LABELS)
            self.assertEqual(len(cli_payload["probabilities"]), len(CLASS_LABELS))

            requests = "\n".join(
                [
                    json.dumps({"op": "ping"}),
                    json.dumps({"op": "reload"}),
                    json.dumps(
                        {
                            "op": "predict",
                            "image": str(data_dir / "paper" / "sample-0.png"),
                        }
                    ),
                    json.dumps(
                        {
                            "op": "predict",
                            "image": str(data_dir / "missing.png"),
                        }
                    ),
                    "{invalid json",
                    json.dumps({"op": "unknown"}),
                ]
            )
            serve_stdout = io.StringIO()
            serve_loop(
                artifacts_dir,
                stdin=io.StringIO(requests + "\n"),
                stdout=serve_stdout,
            )
            responses = [
                json.loads(line) for line in serve_stdout.getvalue().splitlines()
            ]
            self.assertEqual(len(responses), 6)
            self.assertTrue(responses[0]["ok"])
            self.assertTrue(responses[1]["model_loaded"])
            self.assertTrue(responses[2]["ok"])
            self.assertIn(responses[2]["predicted_label"], CLASS_LABELS)
            self.assertFalse(responses[3]["ok"])
            self.assertFalse(responses[4]["ok"])
            self.assertFalse(responses[5]["ok"])


if __name__ == "__main__":
    unittest.main()
