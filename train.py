#!/usr/bin/env python3
"""
Train a 2D object detector on the KITTI dataset in this folder.

This does two things:
  1. Converts the KITTI label files (data_object_label_2/) into YOLO-format
     labels and builds a train/val split under ./yolo_kitti/.
  2. Trains an Ultralytics YOLO detector on that split.

Usage:
    pip install -r requirements.txt
    python train.py                       # convert (if needed) + train
    python train.py --epochs 100 --model yolov8s.pt
    python train.py --skip-prepare        # reuse existing ./yolo_kitti/ split
    python train.py --prepare-only        # just build the YOLO dataset, don't train

The KITTI data provides LiDAR (velodyne) and calibration too; this script only
uses the left color camera images + 2D boxes. That's the accessible, high-value
task. 3D / LiDAR detection would need a different pipeline entirely.
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "data_object_image_2" / "training" / "image_2"
LABEL_DIR = ROOT / "data_object_label_2" / "label_2"
OUT_DIR = ROOT / "yolo_kitti"

# KITTI has 9 label types. "DontCare" marks ignore-regions (not a real object),
# so we drop it. The rest become detector classes.
CLASSES = [
    "Car", "Van", "Truck", "Pedestrian",
    "Person_sitting", "Cyclist", "Tram", "Misc",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
IGNORE = {"DontCare"}


def kitti_to_yolo_line(parts, img_w, img_h):
    """Turn one KITTI label row into a YOLO row, or None if it should be skipped."""
    cls = parts[0]
    if cls in IGNORE or cls not in CLASS_TO_ID:
        return None
    # KITTI: type trunc occ alpha  x1 y1 x2 y2  h w l  tx ty tz  ry
    x1, y1, x2, y2 = (float(parts[4]), float(parts[5]),
                      float(parts[6]), float(parts[7]))
    # Clamp to image bounds (some boxes bleed past the edge).
    x1 = max(0.0, min(x1, img_w)); x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h)); y2 = max(0.0, min(y2, img_h))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def prepare(val_split, seed):
    """Build ./yolo_kitti/{images,labels}/{train,val} + data.yaml."""
    from PIL import Image

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {LABEL_DIR}")

    if OUT_DIR.exists():
        # macOS/Finder can recreate .DS_Store (or ultralytics a .cache) mid-walk,
        # which makes a single rmtree race with "Directory not empty". Retry.
        for _ in range(5):
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            if not OUT_DIR.exists():
                break
        else:
            shutil.rmtree(OUT_DIR)  # last try: surface the real error if any
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    random.shuffle(label_files)
    n_val = int(len(label_files) * val_split)
    val_set = set(label_files[:n_val])

    kept, skipped = 0, 0
    for lf in label_files:
        stem = lf.stem
        img = IMG_DIR / f"{stem}.png"
        if not img.exists():
            skipped += 1
            continue
        split = "val" if lf in val_set else "train"
        w, h = Image.open(img).size

        rows = []
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 8:
                continue
            yolo = kitti_to_yolo_line(p, w, h)
            if yolo:
                rows.append(yolo)

        # Symlink the image (saves ~ a few GB of copies); fall back to copy.
        dst_img = OUT_DIR / "images" / split / f"{stem}.png"
        try:
            dst_img.symlink_to(img)
        except OSError:
            shutil.copy2(img, dst_img)
        (OUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(rows))
        kept += 1

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml = (
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml)
    print(f"Prepared {kept} images ({n_val} val, {kept - n_val} train), "
          f"skipped {skipped} without images.")
    return OUT_DIR / "data.yaml"


def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Base weights, e.g. yolov8n.pt (fast) / yolov8s.pt / yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="cpu, mps, or a CUDA index. Default: auto-detect.")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-prepare", action="store_true",
                    help="Reuse existing ./yolo_kitti/ instead of rebuilding it.")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Only build the YOLO dataset; don't train.")
    args = ap.parse_args()

    data_yaml = OUT_DIR / "data.yaml"
    if not args.skip_prepare:
        data_yaml = prepare(args.val_split, args.seed)
    elif not data_yaml.exists():
        raise SystemExit("--skip-prepare set but ./yolo_kitti/data.yaml doesn't exist. "
                         "Run once without --skip-prepare first.")

    if args.prepare_only:
        print(f"Dataset ready: {data_yaml}")
        return

    from ultralytics import YOLO

    device = args.device or pick_device()
    print(f"Training {args.model} on {device} for {args.epochs} epochs...")
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "runs"),
        name="kitti_yolo",
    )
    print("Done. Best weights + metrics under ./runs/kitti_yolo/")


if __name__ == "__main__":
    main()
