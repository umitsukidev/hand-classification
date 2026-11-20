import io
import json
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import cast

import keras
import numpy as np
import tensorflow as tf

from hand_classification.cli import (
    format_prediction_payload,
    load_classifier,
    load_classifier_bundle,
    main,
    predict_image,
    serve_loop,
)
from hand_classification.data import (
    CLASS_DIRECTORIES,
    IMAGE_SIZE,
    make_image_dataset,
    split_image_paths,
)
from hand_classification.training import save_confusion_matrix, train_model


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((12, 12, 3), color, dtype=np.uint8)
    tf.io.write_file(str(path), tf.io.encode_png(pixels))


def create_dataset(
    root: Path,
    images_per_class: int = 2,
    class_directories: tuple[str, ...] = CLASS_DIRECTORIES,
) -> None:
    colors = ((220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 180, 20))
    for index, directory in enumerate(class_directories):
        color = colors[index % len(colors)]
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
            self.assertEqual(first.class_directories, tuple(sorted(CLASS_DIRECTORIES)))
            self.assertEqual(
                set(first.train_labels), set(range(len(CLASS_DIRECTORIES)))
            )
            self.assertEqual(
                set(first.validation_labels), set(range(len(CLASS_DIRECTORIES)))
            )
            self.assertFalse(set(first.train_paths) & set(first.validation_paths))

    def test_missing_classes_and_too_few_images_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "rock").mkdir()
            with self.assertRaisesRegex(ValueError, "クラスディレクトリ"):
                split_image_paths(data_dir, validation_split=0.2, seed=123)

            create_dataset(
                data_dir,
                images_per_class=1,
                class_directories=("rock", "paper"),
            )
            with self.assertRaisesRegex(ValueError, "2枚以上"):
                split_image_paths(data_dir, validation_split=0.2, seed=123)

    def test_discovers_two_or_more_named_classes_and_keeps_nested_images_in_parent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            class_directories = ("りんご", "alpha", "猫", "beta")
            create_dataset(
                data_dir, images_per_class=2, class_directories=class_directories
            )
            write_image(data_dir / "猫" / "nested" / "extra.png", (80, 80, 80))
            write_image(data_dir / "root-image.png", (1, 2, 3))

            split = split_image_paths(data_dir, validation_split=0.5, seed=123)

            self.assertEqual(split.class_directories, tuple(sorted(class_directories)))
            image_labels = dict(zip(split.train_paths, split.train_labels, strict=True))
            image_labels.update(
                zip(split.validation_paths, split.validation_labels, strict=True)
            )
            self.assertEqual(
                image_labels[data_dir / "猫" / "nested" / "extra.png"],
                split.class_directories.index("猫"),
            )
            self.assertNotIn(data_dir / "root-image.png", image_labels)


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
            create_dataset(
                data_dir,
                class_directories=("zeta", "猫", "alpha", "βeta"),
            )
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

            metadata = json.loads(
                (artifacts_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["class_directories"], list(split.class_directories)
            )
            self.assertEqual(metadata["class_labels"], list(split.class_directories))

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
                    "true_label": split.class_directories[true_label],
                    "predicted_label": split.class_directories[int(predicted_label)],
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
                    "true_label": split.class_directories[true_label],
                    "predicted_label": split.class_directories[int(predicted_label)],
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

            image_path = data_dir / split.class_directories[0] / "sample-0.png"
            label, probabilities = predict_image(
                image_path,
                artifacts_dir,
            )
            self.assertIn(label, split.class_directories)
            self.assertEqual(len(probabilities), len(split.class_directories))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)

            json_stdout = io.StringIO()
            with redirect_stdout(json_stdout):
                main(
                    [
                        "predict",
                        str(image_path),
                        "--artifacts-dir",
                        str(artifacts_dir),
                        "--json",
                    ]
                )
            cli_payload = json.loads(json_stdout.getvalue())
            self.assertTrue(cli_payload["ok"])
            self.assertIn(cli_payload["predicted_label"], split.class_directories)
            self.assertEqual(
                len(cli_payload["probabilities"]), len(split.class_directories)
            )

            requests = "\n".join(
                [
                    json.dumps({"op": "ping"}),
                    json.dumps({"op": "reload"}),
                    json.dumps(
                        {
                            "op": "predict",
                            "image": str(
                                data_dir / split.class_directories[-1] / "sample-0.png"
                            ),
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
            self.assertIn(responses[2]["predicted_label"], split.class_directories)
            self.assertIn(responses[2]["predicted_directory"], split.class_directories)
            self.assertFalse(responses[3]["ok"])
            self.assertFalse(responses[4]["ok"])
            self.assertFalse(responses[5]["ok"])


class ArtifactCompatibilityTests(unittest.TestCase):
    def write_artifact_bundle(
        self,
        artifacts_dir: Path,
        class_directories: tuple[str, ...],
        class_labels: tuple[str, ...],
    ) -> None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(
                    len(class_labels),
                    activation="softmax",
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                ),
            ]
        )
        model.save(artifacts_dir / "hand_classifier.keras")
        (artifacts_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "class_directories": list(class_directories),
                    "class_labels": list(class_labels),
                    "image_size": list(IMAGE_SIZE),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_legacy_japanese_labels_keep_saved_class_order_and_validate_output_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts_dir = root / "artifacts"
            image_path = root / "sample.png"
            write_image(image_path, (20, 40, 60))
            directories = ("rock", "scissors", "paper")
            labels = ("グー", "チョキ", "パー")
            self.write_artifact_bundle(artifacts_dir, directories, labels)

            _, loaded_directories, loaded_labels = load_classifier_bundle(artifacts_dir)
            predicted_label, probabilities = predict_image(image_path, artifacts_dir)
            payload = format_prediction_payload(
                image_path,
                predicted_label,
                probabilities,
                loaded_directories,
                loaded_labels,
            )
            payload_probabilities = cast(
                list[dict[str, object]], payload["probabilities"]
            )

            self.assertEqual(loaded_directories, directories)
            self.assertEqual(loaded_labels, labels)
            self.assertEqual(payload["predicted_directory"], directories[0])
            self.assertEqual(
                [item["directory"] for item in payload_probabilities],
                list(directories),
            )

            metadata_path = artifacts_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["class_directories"] = ["one", "two"]
            metadata["class_labels"] = ["one", "two"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "出力クラス数"):
                load_classifier_bundle(artifacts_dir)

    def test_serve_reloads_model_and_labels_as_one_bundle_when_metadata_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts_dir = root / "artifacts"
            image_path = root / "sample.png"
            write_image(image_path, (20, 40, 60))
            self.write_artifact_bundle(
                artifacts_dir,
                ("class_a", "class_b"),
                ("A", "B"),
            )
            metadata_path = artifacts_dir / "metadata.json"

            class UpdatingInput(io.StringIO):
                def __iter__(self):
                    request = json.dumps({"op": "predict", "image": str(image_path)})
                    yield request + "\n"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["class_directories"] = ["新カテゴリA", "新カテゴリB"]
                    metadata["class_labels"] = ["分類A", "分類B"]
                    metadata_path.write_text(
                        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
                    )
                    os.utime(metadata_path, None)
                    yield request + "\n"

            output = io.StringIO()
            serve_loop(artifacts_dir, stdin=UpdatingInput(), stdout=output)
            responses = [json.loads(line) for line in output.getvalue().splitlines()]

            self.assertEqual(len(responses), 2)
            self.assertEqual(responses[0]["predicted_directory"], "class_a")
            self.assertEqual(responses[0]["predicted_label"], "A")
            self.assertEqual(responses[1]["predicted_directory"], "新カテゴリA")
            self.assertEqual(responses[1]["predicted_label"], "分類A")
            self.assertEqual(
                [entry["directory"] for entry in responses[1]["probabilities"]],
                ["新カテゴリA", "新カテゴリB"],
            )


if __name__ == "__main__":
    unittest.main()
