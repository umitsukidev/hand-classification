import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import keras
import numpy as np
import tensorflow as tf

from hand_classification.data import (
    CLASS_DIRECTORIES,
    CLASS_LABELS,
    IMAGE_SIZE,
    load_image_tensor,
)
from hand_classification.training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hand-classification",
        description="ディレクトリごとの画像分類モデルを学習・実行します。",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="画像フォルダからモデルを学習します"
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("assets/hand_data"),
        help="クラス別サブディレクトリを含むデータフォルダ（既定: assets/hand_data）",
    )
    train_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="モデル・評価結果の保存先（既定: artifacts）",
    )
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--validation-split", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=123)

    predict_parser = commands.add_parser("predict", help="画像1枚のクラスを予測します")
    predict_parser.add_argument("image", type=Path, help="判定する画像ファイル")
    predict_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="学習済みモデルと metadata.json のあるフォルダ（既定: artifacts）",
    )
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="判定結果を JSON 形式で出力します",
    )

    serve_parser = commands.add_parser(
        "serve",
        help="標準入力から JSON リクエストを受け取り推論を実行します",
    )
    serve_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="学習済みモデルと metadata.json のあるフォルダ（既定: artifacts）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "train":
            train_model(
                data_dir=arguments.data_dir,
                artifacts_dir=arguments.artifacts_dir,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                validation_split=arguments.validation_split,
                seed=arguments.seed,
            )
            return

        if arguments.command == "serve":
            serve_loop(arguments.artifacts_dir)
            return

        model, class_directories, class_labels = load_classifier_bundle(
            arguments.artifacts_dir
        )
        predicted_label, probabilities = predict_with_model(
            model, arguments.image, class_labels
        )
        if arguments.json:
            payload = format_prediction_payload(
                arguments.image,
                predicted_label,
                probabilities,
                class_directories,
                class_labels,
            )
            print(json.dumps(payload, ensure_ascii=False))
            return

        predicted_probability = probabilities[class_labels.index(predicted_label)]
        print(f"判定: {predicted_label} ({predicted_probability:.1%})")
        print(
            "確率: "
            + ", ".join(
                f"{label} {probability:.1%}"
                for label, probability in zip(class_labels, probabilities, strict=True)
            )
        )
    except (OSError, ValueError, tf.errors.OpError) as error:
        parser.error(str(error))


def read_artifact_metadata(artifacts_dir: Path) -> dict[str, object]:
    metadata_path = artifacts_dir / "metadata.json"
    if (
        not (artifacts_dir / "hand_classifier.keras").is_file()
        or not metadata_path.is_file()
    ):
        raise ValueError(
            f"モデルまたはラベル情報が見つかりません: {artifacts_dir}"
            "（先に train を実行してください）"
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"ラベル情報を読み取れません: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"ラベル情報の形式が不正です: {metadata_path}")
    class_directories = metadata.get("class_directories")
    class_labels = metadata.get("class_labels")
    if (
        not isinstance(class_directories, list)
        or len(class_directories) < 2
        or not all(isinstance(value, str) and value for value in class_directories)
        or len(set(class_directories)) != len(class_directories)
    ):
        raise ValueError(f"クラスディレクトリ情報の形式が不正です: {metadata_path}")
    if (
        not isinstance(class_labels, list)
        or len(class_labels) != len(class_directories)
        or not all(isinstance(value, str) and value for value in class_labels)
        or len(set(class_labels)) != len(class_labels)
    ):
        raise ValueError(f"クラス名情報の形式が不正です: {metadata_path}")
    if metadata.get("image_size") != list(IMAGE_SIZE):
        raise ValueError(f"モデルの画像サイズが対応していません: {metadata_path}")
    return metadata


def validate_artifacts(artifacts_dir: Path) -> Path:
    read_artifact_metadata(artifacts_dir)
    model_path = artifacts_dir / "hand_classifier.keras"
    return model_path


def load_classifier_bundle(
    artifacts_dir: Path,
) -> tuple[keras.Model, tuple[str, ...], tuple[str, ...]]:
    model_path = validate_artifacts(artifacts_dir)
    metadata = read_artifact_metadata(artifacts_dir)
    model = keras.models.load_model(model_path)
    if not isinstance(model, keras.Model):
        raise ValueError(f"モデルの読み込みに失敗しました: {model_path}")
    raw_directories = metadata["class_directories"]
    raw_labels = metadata["class_labels"]
    if not isinstance(raw_directories, list) or not isinstance(raw_labels, list):
        raise ValueError("モデルのクラス情報が不正です。")
    class_directories = tuple(raw_directories)
    class_labels = tuple(raw_labels)
    output_shape = model.output_shape
    if isinstance(output_shape, list) and len(output_shape) == 1:
        output_shape = output_shape[0]
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or output_shape[-1] != len(class_labels)
    ):
        raise ValueError("モデルの出力クラス数がラベル情報と一致しません。")
    return model, class_directories, class_labels


def load_classifier(artifacts_dir: Path) -> keras.Model:
    model, _, _ = load_classifier_bundle(artifacts_dir)
    return model


def predict_with_model(
    model: keras.Model,
    image_path: Path,
    class_labels: tuple[str, ...] = CLASS_LABELS,
) -> tuple[str, tuple[float, ...]]:
    if not image_path.is_file():
        raise ValueError(f"画像ファイルが見つかりません: {image_path}")

    image = load_image_tensor(image_path)
    prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    if len(prediction) != len(class_labels):
        raise ValueError("モデルの出力クラス数がラベル情報と一致しません。")

    probabilities = tuple(float(value) for value in np.asarray(prediction))
    predicted_label = class_labels[int(np.argmax(prediction))]
    return predicted_label, probabilities


def predict_image(
    image_path: Path,
    artifacts_dir: Path,
) -> tuple[str, tuple[float, ...]]:
    model, _, class_labels = load_classifier_bundle(artifacts_dir)
    return predict_with_model(model, image_path, class_labels)


def format_prediction_payload(
    image_path: Path,
    predicted_label: str,
    probabilities: tuple[float, ...],
    class_directories: tuple[str, ...] = CLASS_DIRECTORIES,
    class_labels: tuple[str, ...] = CLASS_LABELS,
) -> dict[str, object]:
    if len(class_directories) != len(class_labels) or len(probabilities) != len(
        class_labels
    ):
        raise ValueError("クラス情報と予測確率の数が一致しません。")
    predicted_index = class_labels.index(predicted_label)
    return {
        "ok": True,
        "image": str(image_path),
        "predicted_label": predicted_label,
        "predicted_directory": class_directories[predicted_index],
        "probabilities": [
            {
                "directory": directory,
                "label": label,
                "probability": probability,
            }
            for directory, label, probability in zip(
                class_directories,
                class_labels,
                probabilities,
                strict=True,
            )
        ],
    }


def serve_loop(
    default_artifacts_dir: Path,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout

    cached_dir: Path | None = None
    cached_signature: tuple[tuple[int, int], tuple[int, int]] | None = None
    cached_bundle: tuple[keras.Model, tuple[str, ...], tuple[str, ...]] | None = None

    def get_model(
        artifacts_dir: Path,
        force_reload: bool = False,
    ) -> tuple[keras.Model, tuple[str, ...], tuple[str, ...]]:
        nonlocal cached_dir, cached_signature, cached_bundle
        resolved_dir = artifacts_dir.resolve()
        model_path = validate_artifacts(resolved_dir)
        metadata_path = resolved_dir / "metadata.json"
        signature = (
            (model_path.stat().st_mtime_ns, model_path.stat().st_size),
            (metadata_path.stat().st_mtime_ns, metadata_path.stat().st_size),
        )
        if (
            force_reload
            or cached_bundle is None
            or cached_dir != resolved_dir
            or cached_signature != signature
        ):
            cached_bundle = load_classifier_bundle(resolved_dir)
            cached_dir = resolved_dir
            cached_signature = signature
        return cached_bundle

    def write_response(payload: dict[str, object]) -> None:
        output_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output_stream.flush()

    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            write_response({"ok": False, "error": "不正な JSON リクエストです。"})
            continue

        if not isinstance(request, dict):
            write_response(
                {"ok": False, "error": "リクエストは JSON オブジェクトにしてください。"}
            )
            continue

        op = request.get("op")
        raw_artifacts_dir = request.get("artifacts_dir")
        artifacts_dir = (
            Path(raw_artifacts_dir)
            if isinstance(raw_artifacts_dir, str) and raw_artifacts_dir
            else default_artifacts_dir
        )

        try:
            if op == "ping":
                write_response({"ok": True})
            elif op == "reload":
                get_model(artifacts_dir, force_reload=True)
                write_response(
                    {
                        "ok": True,
                        "artifacts_dir": str(artifacts_dir),
                        "model_loaded": True,
                    }
                )
            elif op == "predict":
                raw_image = request.get("image")
                if not isinstance(raw_image, str) or not raw_image:
                    raise ValueError("image パスを指定してください。")
                image_path = Path(raw_image)
                force_reload = bool(request.get("reload", False))
                model, class_directories, class_labels = get_model(
                    artifacts_dir, force_reload=force_reload
                )
                predicted_label, probabilities = predict_with_model(
                    model, image_path, class_labels
                )
                write_response(
                    format_prediction_payload(
                        image_path,
                        predicted_label,
                        probabilities,
                        class_directories,
                        class_labels,
                    )
                )
            else:
                write_response({"ok": False, "error": f"未対応のコマンドです: {op}"})
        except (OSError, ValueError, tf.errors.OpError) as error:
            write_response({"ok": False, "error": str(error)})
