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

        predicted_label, probabilities = predict_image(
            arguments.image,
            arguments.artifacts_dir,
        )
        if arguments.json:
            payload = format_prediction_payload(
                arguments.image,
                predicted_label,
                probabilities,
            )
            print(json.dumps(payload, ensure_ascii=False))
            return

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


def validate_artifacts(artifacts_dir: Path) -> Path:
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
    return model_path


def load_classifier(artifacts_dir: Path) -> keras.Model:
    model_path = validate_artifacts(artifacts_dir)
    model = keras.models.load_model(model_path)
    if not isinstance(model, keras.Model):
        raise ValueError(f"モデルの読み込みに失敗しました: {model_path}")
    return model


def predict_with_model(
    model: keras.Model,
    image_path: Path,
) -> tuple[str, tuple[float, ...]]:
    if not image_path.is_file():
        raise ValueError(f"画像ファイルが見つかりません: {image_path}")

    image = load_image_tensor(image_path)
    prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    if len(prediction) != len(CLASS_LABELS):
        raise ValueError("モデルの出力クラス数がラベル情報と一致しません。")

    probabilities = tuple(float(value) for value in np.asarray(prediction))
    predicted_label = CLASS_LABELS[int(np.argmax(prediction))]
    return predicted_label, probabilities


def predict_image(
    image_path: Path,
    artifacts_dir: Path,
) -> tuple[str, tuple[float, ...]]:
    model = load_classifier(artifacts_dir)
    return predict_with_model(model, image_path)


def format_prediction_payload(
    image_path: Path,
    predicted_label: str,
    probabilities: tuple[float, ...],
) -> dict[str, object]:
    predicted_index = CLASS_LABELS.index(predicted_label)
    return {
        "ok": True,
        "image": str(image_path),
        "predicted_label": predicted_label,
        "predicted_directory": CLASS_DIRECTORIES[predicted_index],
        "probabilities": [
            {
                "directory": directory,
                "label": label,
                "probability": probability,
            }
            for directory, label, probability in zip(
                CLASS_DIRECTORIES,
                CLASS_LABELS,
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
    cached_mtime_ns: int | None = None
    cached_model: keras.Model | None = None

    def get_model(artifacts_dir: Path, force_reload: bool = False) -> keras.Model:
        nonlocal cached_dir, cached_mtime_ns, cached_model
        resolved_dir = artifacts_dir.resolve()
        model_path = validate_artifacts(resolved_dir)
        mtime_ns = model_path.stat().st_mtime_ns
        if (
            force_reload
            or cached_model is None
            or cached_dir != resolved_dir
            or cached_mtime_ns != mtime_ns
        ):
            cached_model = load_classifier(resolved_dir)
            cached_dir = resolved_dir
            cached_mtime_ns = mtime_ns
        return cached_model

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
                model = get_model(artifacts_dir, force_reload=force_reload)
                predicted_label, probabilities = predict_with_model(model, image_path)
                write_response(
                    format_prediction_payload(
                        image_path,
                        predicted_label,
                        probabilities,
                    )
                )
            else:
                write_response({"ok": False, "error": f"未対応のコマンドです: {op}"})
        except (OSError, ValueError, tf.errors.OpError) as error:
            write_response({"ok": False, "error": str(error)})
