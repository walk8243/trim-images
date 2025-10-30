import argparse
import sys
import os
from pathlib import Path
from typing import Iterable, Tuple, Optional

import cv2
import numpy as np


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trim-images",
        description="OpenCV を用いて画像をトリミング/クロップするユーティリティ"
    )

    parser.add_argument(
        "input",
        type=str,
        help="入力パス（画像ファイル or ディレクトリ）"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "center", "rect", "split"],
        help="トリミングモード: auto=余白自動カット, center=アスペクト比でセンタークロップ, rect=指定矩形で切り出し, split=画像を半分に分割"
    )

    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=8,
        help="auto時: 背景色の許容差 (0-255)"
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=0,
        help="auto時: 透明背景扱いのアルファ閾値 (0-255, 0で無効)"
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="auto時: 切り出し後に周囲へ付与するパディング(px)"
    )

    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="center時: アスペクト比（例: 1:1, 16:9, 4:3）"
    )

    # split モード用の分割方向
    parser.add_argument(
        "--split-axis",
        type=str,
        default="v",
        choices=["h", "v", "horizontal", "vertical"],
        help="split時: 分割方向 (h|horizontal=上下に半分, v|vertical=左右に半分)"
    )

    # rect モード用の矩形指定
    parser.add_argument("--x", type=int, default=None, help="rect時: 左上X（px）")
    parser.add_argument("--y", type=int, default=None, help="rect時: 左上Y（px）")
    parser.add_argument("--width", type=int, default=None, help="rect時: 幅（px, >0）")
    parser.add_argument("--height", type=int, default=None, help="rect時: 高さ（px, >0）")

    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="出力拡張子を強制（例: .png, .jpg）。未指定なら元拡張子を維持"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="入力がディレクトリの場合、再帰的に処理"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="同名ファイルが存在する場合に上書きする"
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def find_images(input_path: Path, recursive: bool) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path
        return
    if not input_path.is_dir():
        return
    if recursive:
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in input_path.glob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def read_image_with_alpha(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"画像を読み込めませんでした: {path}")
    return img


def compute_auto_trim_bbox(
    img: np.ndarray,
    bg_threshold: int,
    alpha_threshold: int
) -> Tuple[int, int, int, int]:
    has_alpha = img.shape[2] == 4 if img.ndim == 3 else False

    if has_alpha and alpha_threshold > 0:
        alpha = img[:, :, 3]
        fg_mask = alpha > alpha_threshold
    else:
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        # 周辺の縁ピクセルを背景の代表値とみなして差分を取る
        border = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        bg_val = np.median(border).astype(np.uint8)
        diff = cv2.absdiff(gray, bg_val)
        fg_mask = diff > bg_threshold

    coords = np.column_stack(np.where(fg_mask))
    if coords.size == 0:
        # 全て背景の場合は画像全体
        h, w = img.shape[:2]
        return 0, 0, w, h

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # x, y, w, h へ変換
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def apply_padding(x: int, y: int, w: int, h: int, pad: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x2 = max(0, x - pad)
    y2 = max(0, y - pad)
    w2 = min(width - x2, w + pad * 2)
    h2 = min(height - y2, h + pad * 2)
    return x2, y2, w2, h2


def center_crop_to_aspect(img: np.ndarray, aspect: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    ar_w, ar_h = aspect
    target_ratio = ar_w / ar_h
    cur_ratio = w / h

    if cur_ratio > target_ratio:
        # 横が長い → 幅を削る
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    elif cur_ratio < target_ratio:
        # 縦が長い → 高さを削る
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]
    return img


def crop_by_rect(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if w is None or h is None or x is None or y is None:
        raise ValueError("rect モードには --x, --y, --width, --height の全指定が必要です")
    if w <= 0 or h <= 0:
        raise ValueError("--width と --height は正の整数で指定してください")
    img_h, img_w = img.shape[:2]
    x0 = max(0, min(x, img_w))
    y0 = max(0, min(y, img_h))
    x1 = max(x0, min(x0 + w, img_w))
    y1 = max(y0, min(y0 + h, img_h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("指定矩形が画像範囲外です")
    return img[y0:y1, x0:x1]


def split_image_in_half(img: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray]:
    # axis: 'h' => 水平方向に分割（上下に分かれる）
    #       'v' => 垂直方向に分割（左右に分かれる）
    h, w = img.shape[:2]
    if axis in ("horizontal", "h"):
        # 片側が不足する場合はBORDER_REPLICATEでパディングし、両方同じサイズ（ceil(h/2)）にする
        target_h = (h + 1) // 2  # ceil(h/2)
        top = img[0:target_h, :]
        bottom = img[target_h:h, :]
        need = target_h - bottom.shape[0]
        if need > 0:
            bottom = cv2.copyMakeBorder(bottom, 0, need, 0, 0, cv2.BORDER_REPLICATE)
        return top, bottom
    elif axis in ("vertical", "v"):
        target_w = (w + 1) // 2  # ceil(w/2)
        left = img[:, 0:target_w]
        right = img[:, target_w:w]
        need = target_w - right.shape[1]
        if need > 0:
            right = cv2.copyMakeBorder(right, 0, 0, 0, need, cv2.BORDER_REPLICATE)
        return left, right
    else:
        raise ValueError("axis は 'h'|'horizontal' または 'v'|'vertical' を指定してください")


def parse_aspect(aspect_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not aspect_str:
        return None
    if ":" in aspect_str:
        a, b = aspect_str.split(":", 1)
    else:
        # 例えば "1.7777" のような指定にも一応対応
        try:
            ratio = float(aspect_str)
            # 小数→分数近似（上限小さめ）
            # ここでは簡易的に 1000 を上限に探索
            best = (16, 9)
            best_err = abs(best[0] / best[1] - ratio)
            for q in (1, 2, 3, 4, 5, 8, 9, 10, 16, 100, 125, 500, 1000):
                p = round(ratio * q)
                err = abs(p / q - ratio)
                if err < best_err and p > 0:
                    best = (p, q)
                    best_err = err
            return best
        except ValueError:
            raise ValueError("--aspect は 'W:H' または数値で指定してください")
    try:
        return int(a), int(b)
    except ValueError:
        raise ValueError("--aspect は 'W:H' の整数で指定してください")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_output_path(out_dir: Path, src: Path, base_dir: Path, forced_ext: Optional[str]) -> Path:
    try:
        rel = src.relative_to(base_dir)
    except ValueError:
        rel = src.name
    rel_path = Path(rel)
    file_name = rel_path.stem + (forced_ext if forced_ext else rel_path.suffix)
    return out_dir / rel_path.parent / file_name


def save_image(path: Path, img: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"出力先が既に存在します（--overwriteで上書き可能）: {path}")
    ensure_dir(path.parent)
    # OpenCV はBGR/ BGRAのままでOK
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"保存に失敗しました: {path}")


def process_image_auto(img: np.ndarray, bg_threshold: int, alpha_threshold: int, pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, tw, th = compute_auto_trim_bbox(img, bg_threshold, alpha_threshold)
    if pad > 0:
        x, y, tw, th = apply_padding(x, y, tw, th, pad, w, h)
    return img[y:y + th, x:x + tw]


def process_one(
    src_path: Path,
    mode: str,
    bg_threshold: int,
    alpha_threshold: int,
    pad: int,
    aspect: Optional[Tuple[int, int]],
    rect_params: Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = None,
) -> np.ndarray:
    img = read_image_with_alpha(src_path)
    if mode == "auto":
        return process_image_auto(img, bg_threshold, alpha_threshold, pad)
    if mode == "center":
        if not aspect:
            raise ValueError("center モードには --aspect の指定が必要です")
        return center_crop_to_aspect(img, aspect)
    if mode == "rect":
        if rect_params is None:
            raise ValueError("rect モードの指定が不足しています")
        x, y, w, h = rect_params
        return crop_by_rect(img, int(x), int(y), int(w), int(h))
    raise ValueError(f"未知のモード: {mode}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    out_dir = Path(args.output)
    forced_ext = args.ext
    if forced_ext is not None and not forced_ext.startswith("."):
        forced_ext = "." + forced_ext

    aspect = parse_aspect(args.aspect) if args.mode == "center" else None

    images = list(find_images(input_path, args.recursive))
    if not images:
        print("処理対象の画像が見つかりませんでした", file=sys.stderr)
        return 2

    num_ok = 0
    num_ng = 0
    base_dir = input_path if input_path.is_dir() else input_path.parent
    for src in images:
        try:
            if args.mode == "split":
                img = read_image_with_alpha(src)
                part1, part2 = split_image_in_half(img, args.split_axis)
                base_dst = build_output_path(out_dir, src, base_dir, forced_ext)
                # _1, _2 のサフィックスを付与
                dst1 = base_dst.with_name(base_dst.stem + "_1" + base_dst.suffix)
                dst2 = base_dst.with_name(base_dst.stem + "_2" + base_dst.suffix)
                save_image(dst1, part1, overwrite=args.overwrite)
                save_image(dst2, part2, overwrite=args.overwrite)
                print(f"OK: {src} -> {dst1}, {dst2}")
                num_ok += 1
            else:
                result = process_one(
                    src,
                    mode=args.mode,
                    bg_threshold=args.bg_threshold,
                    alpha_threshold=args.alpha_threshold,
                    pad=args.pad,
                    aspect=aspect,
                    rect_params=(args.x, args.y, args.width, args.height) if args.mode == "rect" else None,
                )
                dst = build_output_path(out_dir, src, base_dir, forced_ext)
                # 保存
                save_image(dst, result, overwrite=args.overwrite)
                print(f"OK: {src} -> {dst}")
                num_ok += 1
        except Exception as e:
            print(f"NG: {src} ({e})", file=sys.stderr)
            num_ng += 1

    print(f"完了: 成功 {num_ok}, 失敗 {num_ng}")
    return 0 if num_ng == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


def cli() -> None:
    raise SystemExit(main())
