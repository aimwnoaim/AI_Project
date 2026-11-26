# Fine-tuning from your recorded videos

This workflow imports **local recorded videos**. No webcam capture, upload server, or cloud storage service is added. Copy/download your recordings onto the computer that will train the model, then give the importer their paths. Videos and extracted frames stay local and are ignored by Git.

## 1. Install

From the repository root in PowerShell, using Python 3.10 or 3.11:

```powershell
git switch dev
git pull --ff-only
cd waste_classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-camera.txt
```

If `.venv` already exists, just activate it and install requirements. OpenCV is needed to decode recorded videos; no camera is opened by these scripts.

## 2. Record and label videos

The exact labels are `plastic`, `metal`, `paper_cardboard`, `other`. Put one physical object of the labeled material in each video. Use your intended camera, background, lighting and distance; vary orientation and include clean/crushed/dirty items. Do not label a whole video plastic if different-material objects enter the scene. Do not import empty setup footage as the object's class.

Give each **physical object a unique item ID**. All recordings of the same bottle must reuse that ID, even across days/backgrounds. Different physical objects need different IDs. Item IDs are global across classes. A video filename can specify the label and identity:

```text
plastic__bottle01__front.mp4
plastic__bottle01__side.mp4
plastic__bottle02__front.mp4
metal__can01__front.mp4
paper_cardboard__box01__front.mp4
other__glass01__front.mp4
```

The optional third component distinguishes clips; it does not change the item group. File extensions MP4, MOV, AVI, MKV and WebM are recognized for directory import, subject to your installed codecs. Use a simple MP4 if decoding fails.

Aim initially for 10–20 **different physical items per class**, with diverse examples, then expand based on errors. Hundreds of nearby frames of one bottle do not replace multiple objects. Preparation requires at least 3 independent item groups per class so every class can appear in train/validation/test; that minimum is only a software check, not evidence of a reliable evaluation.

## 3. Convert recordings to labeled frames

Import a directory of named videos (non-recursive):

```powershell
python -m src.video_data extract --video-dir "C:\waste_videos" --every-seconds 1 --max-frames 150
```

Or label a single video separately, without renaming it:

```powershell
python -m src.video_data extract --video "C:\videos\phone_recording.mp4" --label plastic --item-id bottle01 --every-seconds 1 --max-frames 100
```

For unnamed batches, import each video separately with its physical item ID. Do not assign one shared ID to an entire class unless it really is the same object. An explicit `--label` overrides the filename label; inspect the resulting manifest to catch labeling mistakes.

By default, one frame per second is sampled, capped at 150 frames per clip. The files appear in `data/camera/frames/CLASS/`; `data/camera/frames.csv` records class index, physical item, video/session hash, original frame index, timestamp and image hash. Original videos are never changed. Re-importing identical video bytes is rejected. Imports append to the inventory; use a single importer process at a time. Paths are absolute for reliable local loading: keep the frames in place, or re-import and prepare a new split after moving to another computer.

Review the sampled images and `frames.csv` before splitting. If a video was mislabeled or contains setup footage, remove its rows and corresponding generated frames, then re-import a corrected/trimmed video. Make those corrections before creating the frozen split.

## 4. Prepare independent splits

```powershell
python -m src.video_data prepare --manifest data/camera/frames.csv --output data/camera/split_v1
```

The script validates images and hashes, removes exact duplicate frames, and groups by physical item. If identical frames link two item IDs, those IDs are kept together. Every video/session of an item follows that item's split. Seed 42 assigns approximately 70/15/15 of independent item groups per class to train/validation/test; **frame proportions vary** with clip length. Counts for both frames and items are saved in `statistics.json`.

Existing split directories cannot be overwritten. Keep `split_v1` fixed while comparing experiments. When adding data or correcting labels, deliberately create `split_v2` and document that it is a new evaluation set. Never keep regenerating splits until the test score improves.

The old `data/custom/` dedicated ESP32-CAM test set is separate. This workflow never reads it, trains on it, or merges it into the new camera data.

## 5. Fine-tune and evaluate

Optional one-batch synthetic check (does not modify the saved model):

```powershell
python -m src.finetune_camera --smoke
```

Train using camera training frames:

```powershell
python -m src.finetune_camera --manifest data/camera/split_v1/manifest.csv --run-dir outputs/camera_runs/v1
```

If the original RealWaste training images are still available locally, optionally mix **only their existing training split** into camera training to help retain broader examples:

```powershell
python -m src.finetune_camera --manifest data/camera/split_v1/manifest.csv --base-manifest data/processed/manifest.csv --run-dir outputs/camera_runs/v1_mixed
```

Do not pass `--base-manifest` if those image paths are unavailable. Original RealWaste validation/test rows are excluded from training. Mixing is optional; compare validation results on the same frozen camera split.

Defaults: start from `models/`, batch 16, at most 15 epochs, Adam learning rate 1e-5, last 25 backbone layers eligible for updates, BatchNormalization frozen, early-stopping patience 4. The saved augmentation/preprocessing remains shared with inference. Class weights come from actual training rows only. For smaller memory, use `--batch 8`. Other parameters:

```powershell
python -m src.finetune_camera --help
```

The starting model is scored on **camera validation** first. Only a checkpoint that exceeds its validation macro F1 replaces it; otherwise the starting model is retained. Validation scores control checkpoint selection and early stopping. The chosen model is then evaluated **once** on camera test frames. No threshold tuning is performed. Independent-process reload verification runs afterward. Use a new `--run-dir` for each experiment; existing run folders are never overwritten. Ctrl+C saves an interrupted model and does not run test evaluation. Failed/incomplete runs should be preserved for diagnosis; use a new run name when retrying.

## 6. Read the results

Each run contains:

```text
models/waste_mobilenetv2.keras
models/class_names.json
models/model_metadata.json
camera_manifest.csv
training_manifest.csv
logs/history.csv
logs/run_summary.json
reports/camera_test_metrics.json
reports/camera_test_predictions.csv
reports/classification_report.csv
reports/confusion_matrix.csv
reports/reload_verification.json
plots/accuracy.png
plots/loss.png
plots/confusion_matrix.png
```

Check macro F1, class recall, the confusion matrix, coverage and accuracy among predictions accepted at the existing threshold. Item-level splitting reduces leakage but does not establish robustness to every new camera, material or environment. Softmax confidence is uncalibrated. Do not change the threshold or tune the model based on test errors; use validation data and reserve a fresh independent test collection for later final assessment.

## 7. Use the new model

Keep the original published model unchanged. Select the new model directory explicitly:

```powershell
python -m src.predict --model-dir outputs/camera_runs/v1/models --image "C:\photos\new_item.jpg"
```

The existing optional live preview also accepts the new model:

```powershell
python -m src.webcam --model-dir outputs/camera_runs/v1/models
```

For a future service, call `predict(image_path, model_dir="outputs/camera_runs/v1/models")`. Model directories and metadata have the same interface as the original model. Camera runs are Git-ignored by default because their manifests contain local paths and their models can be large. Review selected run artifacts before explicitly publishing them; do not commit private recordings or frames.

## Current verification status

These tools are supplied for your recordings. No real camera-video training result is claimed until you import your labeled data and run the workflow. The repository's published RealWaste model and historical metrics remain unchanged.

Verified during development: 13 unit tests passed, the synthetic fine-tuning batch passed, and the one-epoch synthetic video-to-export/evaluation workflow passed, including independent-process reload. Synthetic accuracy figures in its log describe random test fixtures, not real-world performance.

Run the synthetic tests with `python -m unittest discover -s tests -v`. The optional `python -m tests.integration_video_workflow` runs a one-epoch synthetic integration check in a temporary directory, including frame extraction, split preparation, export, evaluation and independent-process reload. Synthetic scores are not real-camera performance results.
