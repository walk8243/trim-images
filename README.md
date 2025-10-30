## 概要

OpenCV（opencv-python）を用いて、画像の自動トリミング（余白カット）やアスペクト比センタークロップを行うCLIツールです。

## セットアップ（uv）

このリポジトリは `uv` で管理しています。まず `uv` を用意してください（インストール方法は公式ドキュメント参照）。

依存関係の同期:

```bash
uv sync
```

仮想環境内での実行:

```bash
uv run python --version
```

## 使い方

基本構文:

```bash
uv run trim-images <入力パス> -o <出力ディレクトリ> [オプション]
```

または、ソースから直接:

```bash
uv run python main.py <入力パス> -o <出力ディレクトリ> [オプション]
```

### モード

- `--mode auto`（デフォルト）: 画像の背景色（周縁の代表値）やアルファを元に前景領域を推定し、余白を自動でトリミングします。
  - `--bg-threshold <int>`: 背景とみなす明度差の閾値（0-255, 既定 8）
  - `--alpha-threshold <int>`: アルファがこの値以下なら背景扱い（0-255, 0で無効）
  - `--pad <int>`: 切り出し後のパディング（px）

- `--mode center`: 指定アスペクト比でセンタークロップします。
  - `--aspect <W:H|数値>`: 例 `1:1`, `16:9`, `4:3`。数値（例 `1.777`）も可。

- `--mode rect`: 指定した矩形領域で切り出します。
  - `--x <int>`: 左上X（px）
  - `--y <int>`: 左上Y（px）
  - `--width <int>`: 幅（px, >0）
  - `--height <int>`: 高さ（px, >0）
  - 画像外にかかる場合は自動的に画像範囲へクランプされます。

### 入出力

- 入力は単一ファイルまたはディレクトリを指定可能。
- ディレクトリ指定時は `--recursive` で再帰処理。
- 出力拡張子を強制する場合は `--ext .png` のように指定。
- 既存ファイル上書きは `--overwrite` を指定。

### 例

単一画像を自動トリミングして出力:

```bash
uv run trim-images input.png -o out
```

ディレクトリを再帰処理し、透明背景を考慮、8px のパディングを付与:

```bash
uv run trim-images ./images -o ./out --recursive --alpha-threshold 1 --pad 8
```

16:9 でセンタークロップ:

```bash
uv run trim-images poster.jpg -o out --mode center --aspect 16:9
```

出力拡張子を .png に統一:

```bash
uv run trim-images ./images -o ./out --recursive --ext .png
```

指定位置・サイズでのトリミング（左上 100,50 / 幅 640 / 高さ 360）:

```bash
uv run trim-images input.png -o out --mode rect --x 100 --y 50 --width 640 --height 360
```

## 注意

- 本ツールは OpenCV を用いています。ユーザー要望の「openuv」はおそらく「OpenCV」の誤記と解釈しています。
- 余白検出は周縁の画素から代表背景値を推定するため、画像全体がほぼ一様色の場合は全体が前景と判定されることがあります。

