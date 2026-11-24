"""Single-image CLI and cached model loading for future service integration."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from .config import CLASS_NAMES, CONFIDENCE_THRESHOLD
from .utils import preprocess

def format_prediction(probabilities,threshold=.8):
    values=np.asarray(probabilities,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.any(values<0) or not np.isclose(values.sum(),1,atol=1e-4):
        raise ValueError('Invalid probabilities')
    if not 0<=threshold<=1: raise ValueError('Threshold must be between zero and one')
    index=int(values.argmax()); score=float(values[index]); accepted=score>=threshold
    return dict(label=CLASS_NAMES[index] if accepted else 'retry',confidence=score,should_sort=accepted,probabilities=dict(zip(CLASS_NAMES,values.tolist())))

@lru_cache(maxsize=2)
def load_predictor(model_dir='models'):
    root=Path(model_dir)
    metadata=json.loads((root/'model_metadata.json').read_text())
    if metadata.get('class_names')!=CLASS_NAMES or json.loads((root/'class_names.json').read_text())!=CLASS_NAMES:
        raise ValueError('Invalid class metadata')
    if metadata.get('input_size')!=[224,224,3]: raise ValueError('Invalid input metadata')
    model=tf.keras.models.load_model(root/'waste_mobilenetv2.keras',compile=False)
    if model.output_shape[-1]!=4: raise ValueError('Invalid output shape')
    return model,metadata

def predict(image,model_dir='models',threshold=None):
    path=Path(image)
    if not path.is_file(): raise FileNotFoundError(path)
    if path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}: raise ValueError('Unsupported image format')
    with Image.open(path) as im: im.verify()
    model,metadata=load_predictor(str(model_dir))
    probabilities=model(preprocess(str(path))[None,...],training=False).numpy()[0]
    return format_prediction(probabilities,metadata['confidence_threshold'] if threshold is None else threshold)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--image',required=True); p.add_argument('--model-dir',default='models'); p.add_argument('--threshold',type=float)
    a=p.parse_args()
    try: print(json.dumps(predict(a.image,a.model_dir,a.threshold)))
    except (OSError,ValueError,KeyError) as exc: p.exit(1,f'Prediction failed: {exc}\n')
if __name__=='__main__': main()
