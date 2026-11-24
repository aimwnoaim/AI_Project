# Pipeline commands and limitations

Run all commands from `waste_classifier` with Python 3.10 or 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.download_dataset
python -m src.prepare_dataset
python -m unittest discover -s tests -v
python -m src.train --smoke
python -m src.train
python -m src.verify_export
python -m src.evaluate
python -m src.predict --image sample.jpg
```

Local archives or extracted folders:

```powershell
python -m src.download_dataset --realwaste-path REALWASTE.zip --trashnet-path TRASHNET.zip
python -m src.prepare_dataset --realwaste-path data/raw/realwaste --trashnet-path data/raw/trashnet
```

Official downloads: RealWaste's Download button at https://archive.ics.uci.edu/dataset/908/realwaste and TrashNet's Download raw file at https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip. This run uses RealWaste only, as authorized by the user. TrashNet support is optional; it was excluded after download failures.

The fixed output order is plastic, metal, paper_cardboard, other. The original paper/cardboard categories merge; all remaining categories map to other. Original categories and sources remain in the manifest. Exact duplicate hashes and transitive item/session IDs are grouped across splits. Supply optional metadata with `--metadata metadata.csv` (columns: path, item_id, session_id); paths must match the manifest's relative paths. Grouped stratification targets 70/15/15 and can differ slightly to preserve groups. Conflicting-label duplicates are rejected. Class weights use only training counts.

Neither source supplies physical-item IDs. TrashNet includes multiple views of objects; image splits do not establish generalization to unseen physical objects. The heterogeneous other class cannot cover all real-world objects. Softmax confidence is uncalibrated; unseen items can receive confident wrong predictions.

Add independent ESP32-CAM images in `data/custom/{plastic,metal,paper_cardboard,other}/` before preparation. They become a separate custom test manifest, never training/tuning data. Do not tune on this dedicated custom test set. Metadata is not needed to split this already held-out set.

Input is RGB, aspect-preserving resize with padding to 224x224, raw float pixels 0..255. The saved model contains official MobileNetV2 preprocessing exactly once and training-only augmentation. Do not preprocess again in a future service. Use `src.predict.predict(image)`; model loading is cached. Scores below the configurable 0.80 threshold return retry; confident other returns should_sort=true.

Training defaults: 15 head epochs, batch 32, Adam 1e-3; 20 fine-tuning epochs, last 25 backbone layers with BatchNormalization frozen, Adam 1e-5. The backbone always runs with training=False. Early stopping, checkpoints, learning-rate reduction, and CSV logs are enabled. Best validation accuracy chooses the global best checkpoint across stages. Configure with `python -m src.train --help`. GPU OOM restarts the stage at batch 16 then 8; Ctrl+C saves interrupted state. Native modern Windows TensorFlow uses CPU; use WSL2 or Colab for supported GPU execution. Verify devices with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.

Test evaluation runs once and refuses to overwrite an existing test report. Reports include macro F1, accuracy, per-class scores, source breakdown, original other-category confusions, and acceptance coverage. No threshold tuning is performed.

Common errors: missing dataset folders means the archive path/extraction is wrong; unreadable images appear in rejection statistics; missing model means training/export has not completed; missing or inconsistent metadata prevents prediction. Failed downloads require manual retrieval from the official URLs. Do not commit raw datasets, custom camera images, virtual environments, caches, or secrets.

Model files are generated only after actual training. `models/waste_mobilenetv2.keras`, `class_names.json`, and `model_metadata.json` form the service interface. Import `src.utils` before direct Keras loading to register the preprocessing layer, or use the cached predictor.
