# CarCV — Autonomous Driving Perception Pipeline

A computer vision pipeline for autonomous driving perception on the [KITTI](https://www.cvlibs.net/datasets/kitti/) benchmark. Currently implements object detection via a fine-tuned YOLO11x model. Multi-object tracking (ByteTrack/BoT-SORT) and monocular depth fusion are planned next stages.

## Status

- Object detection (YOLO11x fine-tuned on KITTI) — complete, 0.93 mAP@0.5
- Multi-object tracking (ByteTrack / BoT-SORT) — not yet implemented
- Monocular depth estimation (Depth Anything V2 / ZoeDepth) — not yet implemented
- 2D + depth to 3D fusion and evaluation against KITTI 3D ground truth — not yet implemented

## Expected Repo structure

CarCV/
├── data_object_image_2/ # KITTI object detection images (empty in repo, see below)
├── data_object_label_2/ # KITTI object detection labels (empty in repo, see below)
├── data_object_calib/ # KITTI object detection calibration (empty in repo, see below)
├── data_object_velodyne/ # KITTI object detection LiDAR (empty in repo, see below)
├── data_tracking_image_2/ # KITTI tracking video frames (empty in repo, see below)
├── data_tracking_label_2/ # KITTI tracking labels (empty in repo, see below)
├── data_tracking_calib/ # KITTI tracking calibration (empty in repo, see below)
├── devkit/ # KITTI tracking dev kit / eval code (empty in repo, see below)
├── devkit_object/ # KITTI object detection dev kit (empty in repo, see below)
├── yolo_kitti/ # Generated YOLO-format dataset (built by prepare(), not committed)
├── runs/ # Training run outputs, weights, logs (not committed)
├── train.ipynb # Main training notebook (Colab-ready)
└── README.md




## 1. Getting the data

All KITTI data comes from the official benchmark suite: https://www.cvlibs.net/datasets/kitti/

### Object detection benchmark

Go to `eval_object.php` on the KITTI site (registration required, free, just an email form) and download:

| File | Goes in |
|---|---|
| Left color images of object data set | `data_object_image_2/` |
| Training labels of object data set | `data_object_label_2/` |
| Camera calibration matrices of object data set | `data_object_calib/` |
| Velodyne point clouds | `data_object_velodyne/` |
| Object development kit | `devkit_object/` |

### Tracking benchmark

Go to `eval_tracking.php` on the KITTI site and download:

| File | Goes in |
|---|---|
| Left color images of tracking data set | `data_tracking_image_2/` |
| Training labels of tracking data set | `data_tracking_label_2/` |
| Camera calibration matrices of tracking data set | `data_tracking_calib/` |
| Tracking development kit | `devkit/` |

Not needed for this pipeline (safe to skip): right/stereo images, temporally preceding frames, GPS/IMU data, L-SVM/Regionlet reference detections.

### Alternative: mirror without registration

If the official registration form is unavailable, a full mirror is hosted on DagsHub (no signup required):

```python
from dagshub.streaming import DagsHubFilesystem
fs = DagsHubFilesystem(".", repo_url="https://dagshub.com/DagsHub-Datasets/kitti-dataset")
```

### Expected extracted structure

After unzipping, you should have:

## 2. Where to store it

- Local / Colab disk: fastest for training. Colab resets on runtime disconnect.
- Google Drive: persists across sessions but slower I/O, especially for many small files. Recommended pattern: keep the `.zip` files on Drive, copy them to local Colab disk each session, and extract locally:

```python
import shutil
shutil.copy2('/content/drive/MyDrive/kitti_zips/data_object_image_2.zip', '/content/kitti/')
```

then unzip locally before training.

## 3. Setup

```bash
pip install ultralytics
```

If running in Colab, mount Drive first if your data/weights live there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. Training

Open `CarVisual.ipynb`. Set the config cell at the top:

```python
ROOT = Path("/content/kitti")   # wherever your extracted KITTI folders live
MODEL = "yolo11x.pt"             # auto-downloaded from Ultralytics on first run
EPOCHS = 50                    #Change to a higher number for better results
IMGSZ = 640
BATCH = 16
```

Run cells in order:

1. Prepare — converts KITTI labels to YOLO format, builds `yolo_kitti/` train/val split
2. Sanity check — spot-checks one converted image/label pair
3. Device check — confirms GPU is available
4. Train — fine-tunes YOLO11x via `model.train(...)`

Best weights land at `runs/kitti_yolo11x/weights/best.pt`.

### Continuing training

To train additional epochs from a completed run:

```python
model = YOLO(ROOT / "runs" / "kitti_yolo11x" / "weights" / "best.pt")
model.train(data=str(data_yaml), epochs=50, imgsz=IMGSZ, batch=BATCH,
            device=device, project=str(ROOT / "runs"), name="kitti_yolo11x_continued")
```

## 5. Pre-trained models

- Base YOLO11 weights (before any fine-tuning) are auto-downloaded by Ultralytics the first time you instantiate `YOLO("yolo11x.pt")`, sourced from Ultralytics' official releases: https://github.com/ultralytics/assets/releases
- This project's fine-tuned checkpoint (`best.pt`, 0.93 mAP@0.5 on KITTI) is not committed to git due to file size. [Add hosting link here once uploaded, e.g. Hugging Face Hub or a GitHub Release.]

## 6. Evaluation

- mAP / precision / recall: reported automatically by `model.val()` during and after training.
- mIoU: custom greedy-matching IoU script included in the notebook (`compute_miou`), reported per-class and overall.
- Loss curves: plotted from `runs/<name>/results.csv` via the plotting cell in the notebook.

## Future Work

- Add ByteTrack/BoT-SORT tracking on top of detections, evaluate with MOTA/HOTA using TrackEval
- Add monocular depth estimation (Depth Anything V2), validate against KITTI LiDAR ground truth
- Fuse 2D tracks and depth into 3D trajectories, evaluate 3D IoU against KITTI 3D boxes
- Failure case analysis (occlusion, distance, class imbalance)
