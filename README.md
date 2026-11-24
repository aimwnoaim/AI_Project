# Camera waste classifier

A TensorFlow/Keras MobileNetV2 pipeline for a camera-based waste sorter. Images are classified in this fixed order:

| Index | Class | RealWaste categories |
|---|---|---|
| 0 | `plastic` | Plastic |
| 1 | `metal` | Metal |
| 2 | `paper_cardboard` | Paper, Cardboard |
| 3 | `other` | Food Organics, Glass, Miscellaneous Trash, Textile Trash, Vegetation |

The current experiment uses **RealWaste only**. TrashNet acquisition support remains optional. This repository contains the ML pipeline; ESP32 firmware, web serving, and motor control are outside its scope.

## Status

Source code is published on `dev`. Training and runtime verification are in progress. A trained model and measured results must not be assumed to exist until the files listed below are present. No accuracy claim is made here.

## Quick start

Use Python **3.10 or 3.11**. On Windows, TensorFlow runs on CPU; use a supported WSL2/Colab environment for GPU training.

```powershell
git clone --branch dev https://github.com/aimwnoaim/AI_Project.git
cd AI_Project/waste_classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_pipeline
```

On Linux/macOS, activate with `source .venv/bin/activate`. Training downloads ImageNet backbone weights on first use. CPU training can take considerable time.

Already have RealWaste downloaded?

```powershell
python -m src.run_pipeline --realwaste-path "C:\datasets\realwaste.zip"
```

The command validates/prepares the dataset, runs synthetic unit tests, runs a small training smoke test, trains both stages with early stopping, checks export reload in another process, then evaluates the untouched test set. It stops on the first failure. It does not automatically commit or push.

## Predict after training

From `waste_classifier`:

```powershell
python -m src.predict --image "C:\images\bottle.jpg"
```

The JSON response includes `label`, real maximum `confidence`, `should_sort`, and all four probabilities. At the default threshold 0.80, lower-confidence predictions return `retry`. A confident `other` prediction is accepted. Scores are uncalibrated and are not a guarantee of correctness.

For a future Python service, use `from src.predict import predict` and call `predict(image_path)`. The model is loaded once and cached. No Flask service is included.

## Project layout

```text
waste_classifier/
  src/                  acquisition, preparation, training, evaluation, inference
  tests/                synthetic-image unit tests
  data/raw/             downloaded/extracted images (ignored)
  data/processed/       reproducible manifests and dataset statistics
  data/custom/          held-out camera images (ignored)
  models/               trained Keras model and interface metadata
  outputs/plots/        learning curves and confusion matrices
  outputs/reports/      test metrics, predictions and export verification
  outputs/logs/         environment, epoch histories and acquisition records
  requirements.txt      pinned main dependencies
```

## Artifacts produced by a successful run

- `models/waste_mobilenetv2.keras`, `class_names.json`, `model_metadata.json`
- `data/processed/manifest.csv`, `statistics.json`
- `outputs/logs/history.csv`, stage logs, environment information
- `outputs/reports/test_metrics.json`, predictions CSV, classification report and confusion matrix CSV
- `outputs/plots/accuracy.png`, `loss.png`, `test_confusion_matrix.png`
- `outputs/reports/reload_verification.json`

The headline evaluation metric is **macro F1**, accompanied by accuracy and per-class precision/recall. Exact duplicate hashes and supplied item/session metadata are kept together. Physical-item independence cannot be guaranteed without reliable IDs. Results on the public dataset do not establish performance under your bin's camera and lighting.

See the [pipeline guide](waste_classifier/README.md) for individual commands, hyperparameters, metadata, camera test images, and troubleshooting.

## Data attribution

[RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste): Single, S., Iranmanesh, S., & Raad, R. (2023), UCI Machine Learning Repository, [DOI 10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G), CC BY 4.0. Dataset images are not redistributed here. Optional [TrashNet](https://github.com/garythung/trashnet) is by Gary Thung and Mindy Yang; consult its original repository for licensing and attribution.
