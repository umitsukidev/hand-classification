import argparse
import json
from collections.abc import Sequence
from pathlib import Path

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
        description="グー・チョキ・パーの画像分類モデルを学習・実行します。",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="画像フォルダからモデルを学習します"
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/assets/hand_data"),
        help="rock/scissors/paper の画像フォルダ（既定: src/assets/hand_data）",
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

        predicted_label, probabilities = predict_image(
            arguments.image,
            arguments.artifacts_dir,
        )
        predicted_probability = probabilities[CLASS_LABELS.index(predicted_label)]
        print(f"判定: {predicted_label} ({predicted_probability:.1%})")
        print(
            "確率: "
            + ", ".join(
                f"{label} {probability:.1%}"
                for label, probability in zip(CLASS_LABELS, probabilities, strict=True)
            )
        )
    except (OSError, ValueError, tf.errors.OpError) as error:
        parser.error(str(error))


def predict_image(
    image_path: Path,
    artifacts_dir: Path,
) -> tuple[str, tuple[float, ...]]:
    if not image_path.is_file():
        raise ValueError(f"画像ファイルが見つかりません: {image_path}")

    model_path = artifacts_dir / "hand_classifier.keras"
    metadata_path = artifacts_dir / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
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
    if metadata.get("class_directories") != list(CLASS_DIRECTORIES):
        raise ValueError(f"モデルのクラス順が対応していません: {metadata_path}")
    if metadata.get("class_labels") != list(CLASS_LABELS):
        raise ValueError(f"モデルのクラス名が対応していません: {metadata_path}")
    if metadata.get("image_size") != list(IMAGE_SIZE):
        raise ValueError(f"モデルの画像サイズが対応していません: {metadata_path}")

    model = keras.models.load_model(model_path)
    image = load_image_tensor(image_path)
    prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    if len(prediction) != len(CLASS_LABELS):
        raise ValueError("モデルの出力クラス数がラベル情報と一致しません。")

    probabilities = tuple(float(value) for value in np.asarray(prediction))
    predicted_label = CLASS_LABELS[int(np.argmax(prediction))]
    return predicted_label, probabilities
