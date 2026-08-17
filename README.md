# CarCV — Autonomous Driving Perception Pipeline

End-to-end perception pipeline on the KITTI autonomous driving benchmark: object detection (YOLO11x), multi-object tracking (ByteTrack), and monocular depth-based 3D localization (Depth Anything V2).

**Current results:** 0.93 mAP@0.5 (detection) · 0.858 MOTA across 20 KITTI tracking sequences · <2m median 3D position error within 20m, validated against LiDAR ground truth over 400+ instances.

## 1. Dataset

Uses the KITTI Object Detection and Tracking benchmarks:
- `data_object_image_2.zip` — left color images (detection)
- `data_object_label_2.zip` — detection labels
- `data_tracking_image_2.zip` — tracking sequences
- `data_tracking_label_2.zip` — tracking labels

Download from the [KITTI website](https://www.cvlibs.net/datasets/kitti/) and store the zips in Google Drive (`/content/drive/MyDrive/kitti_zips/`) if working in Colab.

## 2. Data Prep (Google Drive → Colab)

Google Drive's FUSE mount is too slow to read directly from during training and will starve the GPU. Copy zips to local Colab disk each session, then extract locally:

```python
import shutil
shutil.copy2('/content/drive/MyDrive/kitti_zips/data_object_image_2.zip', '/content/kitti/')
shutil.copy2('/content/drive/MyDrive/kitti_zips/data_tracking_image_2.zip', '/content/kitti/')
```

Unzip locally before training:

```python
import zipfile
for z in Path("/content/kitti").glob("*.zip"):
    with zipfile.ZipFile(z) as f:
        f.extractall("/content/kitti")
```

## 3. Setup

```bash
pip install ultralytics
```

If running in Colab, mount Drive first if your data/weights live there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. Detection Training

Open `CarVisual.ipynb`. Set the config cell at the top:

```python
ROOT = Path("/content/kitti")   # wherever your extracted KITTI folders live
MODEL = "yolo11x.pt"            # auto-downloaded from Ultralytics on first run
EPOCHS = 50                     # increase for better results
IMGSZ = 640
BATCH = 16
```

Run cells in order:
1. **Prepare** — converts KITTI labels to YOLO format, builds `yolo_kitti/` train/val split
2. **Sanity check** — spot-checks one converted image/label pair
3. **Device check** — confirms GPU is available
4. **Train** — fine-tunes YOLO11x via `model.train(...)`

Best weights land at `runs/kitti_yolo11x/weights/best.pt`.

### Continuing training

To train additional epochs from a completed run:

```python
model = YOLO(ROOT / "runs" / "kitti_yolo11x" / "weights" / "best.pt")
model.train(data=str(data_yaml), epochs=50, imgsz=IMGSZ, batch=BATCH,
            device=device, project=str(ROOT / "runs"), name="kitti_yolo11x_continued")
```

## 5. Pre-trained Models

- Base YOLO11 weights (before any fine-tuning) are auto-downloaded by Ultralytics the first time you instantiate `YOLO("yolo11x.pt")`, sourced from [Ultralytics' official releases](https://github.com/ultralytics/assets/releases).
- This project's fine-tuned checkpoint (`best.pt`, 0.93 mAP@0.5 on KITTI) is not committed to git due to file size. [Add hosting link here once uploaded, e.g. Hugging Face Hub or a GitHub Release.]

## 6. Tracking (ByteTrack)

Detections are fed frame-by-frame into ByteTrack to produce consistent object IDs across each KITTI tracking sequence.

- Run via `Track.ipynb` (or the tracking cell in the main notebook), pointing at the fine-tuned `best.pt` weights and a KITTI tracking sequence folder.
- Evaluated with [TrackEval](https://github.com/JonathonLuiten/TrackEval) using the KITTI tracking ground truth.
- **Result:** 0.858 MOTA across 20 tracking sequences.

## 7. Depth Estimation & 3D Fusion

Monocular depth maps from **Depth Anything V2** are fused with 2D detections/tracks to localize each object in 3D:

1. Run Depth Anything V2 inference on each frame to get a dense depth map.
2. Sample depth at each 2D box's footprint (bottom-center or median-in-box, depending on the notebook cell used) to estimate object distance.
3. Back-project into 3D using the KITTI camera calibration files (`calib/*.txt`).
4. Compare against KITTI LiDAR-derived ground-truth 3D boxes.

- **Result:** under 2m median position error within 20m range, validated across 400+ instances, with error growth characterized as a function of distance.

## 8. Evaluation Summary

| Stage | Metric | Result |
|---|---|---|
| Detection | mAP@0.5 | 0.93 |
| Tracking | MOTA (20 sequences) | 0.858 |
| 3D Localization | Median position error (<20m) | <2m |

- mAP / precision / recall: reported automatically by `model.val()` during and after training.
- mIoU: custom greedy-matching IoU script included in the notebook (`compute_miou`), reported per-class and overall.
- Loss curves: plotted from `runs/<name>/results.csv` via the plotting cell in the notebook.
- MOTA/HOTA: computed via TrackEval against KITTI tracking ground truth.
- 3D position error: computed against KITTI LiDAR ground-truth boxes, binned by distance.

## Future Work

- Extend MOT evaluation to full HOTA breakdown (association vs. detection accuracy) rather than MOTA alone
- Evaluate 3D IoU directly against KITTI 3D boxes (current validation is position-error based)
- Failure case analysis (occlusion, distance, class imbalance)
- Explore BoT-SORT as a tracking alternative to ByteTrack for a head-to-head comparison
- Real-time inference profiling (end-to-end FPS across detection → tracking → depth fusion)
