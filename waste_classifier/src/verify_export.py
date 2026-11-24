"""Verify an exported model in an independent Python process."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from .utils import write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--child',action='store_true'); a=p.parse_args()
    x=np.random.default_rng(42).uniform(0,255,(1,224,224,3)).astype('float32')
    model=tf.keras.models.load_model('models/waste_mobilenetv2.keras',compile=False)
    y=model(x,training=False).numpy()
    path=Path('outputs/reports/reload_child.npy')
    if a.child:
        np.save(path,y); return
    subprocess.run([sys.executable,'-m','src.verify_export','--child'],check=True)
    other=np.load(path); np.testing.assert_allclose(y,other,rtol=1e-5,atol=1e-6)
    write_json('outputs/reports/reload_verification.json',{'passed':True,'max_absolute_difference':float(np.max(np.abs(y-other))),'separate_process':True})
    path.unlink(); print('Separate-process reload verification passed')

if __name__=='__main__': main()
