"""One tiny synthetic integration run; not a real-camera performance experiment."""
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from src.video_data import extract,prepare
from src.config import CLASS_NAMES

with tempfile.TemporaryDirectory(prefix='video_workflow_') as temp:
    root=Path(temp)
    for c,label in enumerate(CLASS_NAMES):
        for item in range(3):
            path=root/f'{label}__{label}{item}__clip.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),4,(32,24))
            if not writer.isOpened(): raise RuntimeError('Synthetic video encoder unavailable')
            for i in range(4): writer.write(np.random.default_rng(c*100+item*10+i).integers(0,255,(24,32,3),dtype=np.uint8))
            writer.release(); extract(path,root/'inventory',max_frames=1)
    prepare(root/'inventory/frames.csv',root/'split')
    subprocess.run([sys.executable,'-m','src.finetune_camera','--manifest',str(root/'split/manifest.csv'),'--run-dir',str(root/'run'),'--epochs','1','--batch','4'],check=True)
    for name in ['models/waste_mobilenetv2.keras','models/model_metadata.json','reports/camera_test_metrics.json','reports/reload_verification.json','logs/history.csv']:
        assert (root/'run'/name).exists(),name
    print('Synthetic video-to-training-to-evaluation integration PASSED; temporary outputs removed.',flush=True)
