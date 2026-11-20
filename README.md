# グー・チョキ・パー画像分類

TensorFlow/Keras の CNN で、画像をグー・チョキ・パーに分類します。学習用の画像は自分で用意してください。自動ダウンロードは行いません。

## セットアップ

Python 3.11 以上と [uv](https://docs.astral.sh/uv/) を用意し、プロジェクトのルートで依存関係をインストールします。GPUI デスクトップ UI を起動する場合は Rust 1.95 以上も必要です。

```bash
uv sync
```

## 画像の配置

画像を次のディレクトリに分けて置きます。フォルダ名とクラスの順序は固定です。

```text
assets/hand_data/
├── rock/      # グー
├── scissors/  # チョキ
└── paper/     # パー
```

各クラスに2枚以上の画像が必要です。対応形式は JPEG、PNG、BMP、GIF です。画像は学習時に読み込まれ、すべて96×96ピクセルのRGBに揃えられます。

## 学習

```bash
uv run hand-classification train
```

デフォルトではクラスごとに乱数seed 123で学習用と検証用へ分割し、最大60 epoch 学習します。検証用画像は各クラスに最低1枚確保されます。早期終了を使い、モデルと評価結果を `artifacts/` に保存します。

設定を変える場合はオプションを指定します。

```bash
uv run hand-classification train \
  --data-dir assets/hand_data \
  --artifacts-dir artifacts \
  --epochs 60 \
  --batch-size 32 \
  --validation-split 0.2 \
  --seed 123
```

保存されるファイルは `hand_classifier.keras`（モデル）、`metadata.json`（クラス順と画像サイズ）、`training_history.png`（正解率の推移）、`confusion_matrix.png`（検証データの混同行列）、`misclassified_images.json`（検証件数、誤分類画像、正しく分類した画像の一覧）です。画像一覧の各項目には、画像パス・正解クラス・予測クラスが含まれます。

## 画像を判定

学習後、画像ファイルを指定します。

```bash
uv run hand-classification predict path/to/hand.jpg
```

別の保存先を使った場合は、同じディレクトリを指定します。

```bash
uv run hand-classification predict path/to/hand.jpg --artifacts-dir artifacts
```

判定したクラスと各クラスの確率を表示します。

## デスクトップ UI (GPUI)

GPUI を使ったデスクトップ UI を起動できます。

```bash
cargo run
```

- **画像判定タブ**: 画像ファイルのドラッグ＆ドロップまたはファイル選択から推論を実行し、予測クラス・確率バー・最近の判定履歴を表示します。
- **モデル学習タブ**: `data-dir` / `artifacts-dir` や Epochs・Batch Size・Validation Split・Seed を設定して学習を実行し、リアルタイムログと評価グラフ（`training_history.png` / `confusion_matrix.png`）を確認できます。誤分類画像はサムネイルのギャラリーから拡大表示できます。混同行列をクリックするとセル別の検証画像を表示し、そこから各画像も拡大できます。セル別画像のない古い学習結果は、再学習すると利用できます。

## 構成

```text
.
├── apps/
│   └── ui/                       # GPUI デスクトップ UI (Rust / Cargo workspace member)
└── packages/
    └── hand-classification/      # TensorFlow/Keras 分類パッケージ (Python / uv workspace member)
```

## 開発チェック

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python -m unittest discover -s packages/hand-classification/tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

表示される正解率と混同行列は、学習時に分けた検証データでの結果です。独立したテストデータでの評価ではなく、実データに対する精度も確認していません。
