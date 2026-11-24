# Waste classifier — acquisition blocked

This is incomplete setup work, not a trained pipeline. No training, evaluation, or model export has run.

Fixed class order: `plastic`, `metal`, `paper_cardboard`, `other`.

## Environment

Python 3.11.9 was downloaded into the parent workspace's `work/python311`. The system Python is 3.13.7. Dependency installation encountered repeated network timeouts; TensorFlow device detection has not run.

## Dataset blocker

The official TrashNet repository was inspected. Its archive is `data/dataset-resized.zip`. Both the github.com raw redirect and raw.githubusercontent.com download attempts failed with transport errors (connection forcibly closed / response ended prematurely).

Download it manually from https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip using **Download raw file**, and save it as `data/raw/trashnet.zip`. Do not use the incomplete archive left by the failed attempt.

The official RealWaste archive is available from the Download button at https://archive.ics.uci.edu/dataset/908/realwaste. An earlier local archive exists outside this repository. The ucimlrepo image-fetch check has not run because dependency setup is incomplete.

## Recovery commands

From this directory, using Python 3.11:

```powershell
python -m pip install -r requirements.txt
python -m src.download_dataset --realwaste-path "PATH_TO_REALWASTE_ZIP" --trashnet-path "PATH_TO_TRASHNET_ZIP"
```

The acquisition helper blocks ZIP path traversal and accepts local folders. Full dataset validation, preparation, training, evaluation, inference, and tests remain unimplemented. No metrics are claimed.

Publication target: `dev` in https://github.com/aimwnoaim/AI_Project. The repository was empty at inspection, so `dev` was created as an orphan branch; main was not modified. These setup files are committed locally only.
