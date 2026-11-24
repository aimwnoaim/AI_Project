"""Local webcam preview with live four-class predictions; Q/Esc exits."""
import argparse
import time
from .predict import load_predictor, format_prediction
from .utils import preprocess_rgb

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--model-dir',default='models')
    parser.add_argument('--threshold',type=float)
    args=parser.parse_args()
    try:
        import cv2
    except ImportError:
        parser.exit(1,'Install camera support: python -m pip install -r requirements-camera.txt\n')
    model,metadata=load_predictor(args.model_dir)
    threshold=metadata['confidence_threshold'] if args.threshold is None else args.threshold
    if not 0<=threshold<=1: parser.error('Threshold must be between 0 and 1')
    camera=cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps, check camera permissions, or try --camera 1.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        while True:
            ok,frame=camera.read()
            if not ok: raise RuntimeError('Camera stopped returning frames')
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            start=time.perf_counter()
            probabilities=model(preprocess_rgb(rgb)[None,...],training=False).numpy()[0]
            elapsed=time.perf_counter()-start
            result=format_prediction(probabilities,threshold)
            color=(0,220,0) if result['should_sort'] else (0,180,255)
            title=f"{result['label']}  {result['confidence']:.1%}  {elapsed*1000:.0f} ms"
            cv2.putText(frame,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            cv2.putText(frame,'One item at a time | Q/Esc: exit',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            for i,(name,probability) in enumerate(result['probabilities'].items()):
                cv2.putText(frame,f'{name}: {probability:.1%}',(12,85+i*23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imshow('Waste classifier - local camera',frame)
            if cv2.waitKey(1)&0xff in (ord('q'),27): break
            if cv2.getWindowProperty('Waste classifier - local camera',cv2.WND_PROP_VISIBLE)<1: break
    except (RuntimeError,KeyboardInterrupt) as exc:
        if isinstance(exc,RuntimeError): parser.exit(1,str(exc)+'\n')
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__=='__main__': main()
