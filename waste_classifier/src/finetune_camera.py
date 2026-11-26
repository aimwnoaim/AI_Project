"""Fine-tune a saved classifier on grouped camera data; export into a new run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import CLASS_NAMES
from .utils import dataset, environment, write_json
from .predict import load_predictor
from .evaluate import metrics

def validate_manifest(frame):
    required={'path','label','class_index','split','item_id','group','sha256','source'}
    if not required.issubset(frame): raise ValueError('Use video_data prepare to create the camera manifest first')
    if set(frame.source)!={'camera_video'}: raise ValueError('Only a prepared camera-video manifest is accepted')
    if set(frame.split)!={'train','val','test'}: raise ValueError('Manifest must contain train/val/test')
    for field in ['group','item_id','sha256']:
        if frame.groupby(field).split.nunique().max()>1: raise ValueError(f'{field} crosses splits')
    for row in frame.itertuples():
        if row.label not in CLASS_NAMES or row.class_index!=CLASS_NAMES.index(row.label): raise ValueError('Invalid labels')
    for split in ['train','val','test']:
        if set(frame.loc[frame.split==split,'label'])!=set(CLASS_NAMES): raise ValueError(f'{split} must contain all four classes')

class ValidationF1(tf.keras.callbacks.Callback):
    def __init__(self,frame,batch): super().__init__(); self.frame=frame; self.batch=batch
    def on_epoch_end(self,epoch,logs=None):
        pred=self.model.predict(dataset(self.frame,self.batch),verbose=0).argmax(1)
        logs['val_macro_f1']=f1_score(self.frame.class_index,pred,labels=range(4),average='macro',zero_division=0)
        print(f"Validation macro F1: {logs['val_macro_f1']:.4f}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='data/camera/split_v1/manifest.csv'); p.add_argument('--base-model-dir',default='models'); p.add_argument('--base-manifest',help='Optional original dataset manifest; only its training rows are mixed in')
    p.add_argument('--run-dir',default='outputs/camera_runs/v1'); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch',type=int,default=16); p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--fine-layers',type=int,default=25); p.add_argument('--patience',type=int,default=4)
    p.add_argument('--smoke',action='store_true',help='Synthetic one-batch check only; no camera dataset required')
    a=p.parse_args()
    if min(a.epochs,a.batch,a.lr,a.fine_layers,a.patience)<=0: p.error('Training parameters must be positive')
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    env=environment(); model,metadata=load_predictor(a.base_model_dir)
    backbone=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model) and 'mobilenet' in layer.name)
    backbone.trainable=True
    for layer in backbone.layers: layer.trainable=False
    for layer in backbone.layers[-a.fine_layers:]:
        if not isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    if a.smoke:
        result=model.train_on_batch(np.zeros((2,224,224,3),np.float32),np.array([0,1])); print('Camera fine-tune smoke passed:',result); return
    frame=pd.read_csv(a.manifest,dtype={'item_id':str}); validate_manifest(frame)
    train=frame[frame.split=='train'].copy(); val=frame[frame.split=='val']; test=frame[frame.split=='test']
    # Verify all paths/hashes before using data; never silently reuse changed splits.
    from .video_data import file_hash
    for row in frame.itertuples():
        if file_hash(row.path)!=row.sha256: raise ValueError(f'Frame changed: {row.path}')
    if a.base_manifest:
        original=pd.read_csv(a.base_manifest); original=original[original.split=='train']
        if not set(original.source).issubset({'realwaste','trashnet'}): raise ValueError('Base manifest must contain original public-dataset training rows only')
        for row in original.itertuples():
            if row.label not in CLASS_NAMES or row.class_index!=CLASS_NAMES.index(row.label): raise ValueError('Invalid base-manifest labels')
        if set(original.sha256)&set(frame[frame.split!='train'].sha256): raise ValueError('Original training data overlaps camera holdout')
        train=pd.concat([train,original],ignore_index=True)
        if train.groupby('sha256').label.nunique().max()>1: raise ValueError('Conflicting labels for identical training images')
        train=train.drop_duplicates('sha256')
    run=Path(a.run_dir)
    if run.exists(): raise FileExistsError(f'Run exists: {run}; use a new run name')
    for sub in ['models','reports','plots','logs']: (run/sub).mkdir(parents=True,exist_ok=True)
    frame.to_csv(run/'camera_manifest.csv',index=False); train.to_csv(run/'training_manifest.csv',index=False)
    counts=train.class_index.value_counts(); weights={i:len(train)/(4*int(counts[i])) for i in range(4)}
    best_path=run/'models/waste_mobilenetv2.keras'
    baseline=model.predict(dataset(val,a.batch),verbose=0).argmax(1)
    baseline_f1=float(f1_score(val.class_index,baseline,average='macro',labels=range(4),zero_division=0))
    model.save(best_path)
    checkpoint_template=str(best_path).replace('{','{{').replace('}','}}')
    callbacks=[ValidationF1(val,a.batch),tf.keras.callbacks.ModelCheckpoint(checkpoint_template,monitor='val_macro_f1',mode='max',save_best_only=True,initial_value_threshold=baseline_f1),tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1',mode='max',patience=a.patience,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5),tf.keras.callbacks.CSVLogger(str(run/'logs/history.csv'))]
    try:
        history=model.fit(dataset(train,a.batch,True),validation_data=dataset(val,a.batch),epochs=a.epochs,class_weight=weights,callbacks=callbacks,verbose=2)
    except KeyboardInterrupt:
        model.save(run/'models/interrupted.keras'); print('Interrupted state saved; no test evaluation performed'); raise SystemExit(130)
    chosen=tf.keras.models.load_model(best_path,compile=False)
    best_f1=max(baseline_f1,max(history.history['val_macro_f1']))
    updated=dict(metadata,training_date=datetime.now(timezone.utc).isoformat(),dataset_sources=sorted(train.source.unique()),selected_checkpoint='camera_fine_tuned' if best_f1>baseline_f1 else 'unchanged_baseline',best_validation_macro_f1=best_f1,baseline_validation_macro_f1=baseline_f1,training_config=vars(a),environment=env,tensorflow_version=tf.__version__)
    updated['parent_validation_accuracy']=updated.pop('best_validation_accuracy',None)
    updated['parent_model_sha256']=file_hash(Path(a.base_model_dir)/'waste_mobilenetv2.keras')
    selected_val=chosen.predict(dataset(val,a.batch),verbose=0).argmax(1)
    updated['selected_validation_accuracy']=float(np.mean(selected_val==val.class_index.to_numpy()))
    updated['label_mapping']=list(metadata.get('label_mapping',[]))+[dict(source='camera_video',original_category=label,label=label) for label in CLASS_NAMES]
    write_json(run/'models/model_metadata.json',updated); write_json(run/'models/class_names.json',CLASS_NAMES)
    for metric in ['accuracy','loss']:
        fig,ax=plt.subplots(); ax.plot(history.history[metric],label='train'); ax.plot(history.history['val_'+metric],label='val'); ax.legend(); ax.set(xlabel='Epoch',ylabel=metric); fig.savefig(run/'plots'/f'{metric}.png'); plt.close(fig)
    # Read test predictions once, only after model selection is complete.
    probabilities=chosen.predict(dataset(test,a.batch),verbose=0); result=test.copy(); result['predicted']=probabilities.argmax(1); result['confidence']=probabilities.max(1)
    for i,label in enumerate(CLASS_NAMES): result['prob_'+label]=probabilities[:,i]
    report=metrics(result); accepted=result.confidence>=updated['confidence_threshold']
    report['threshold']={'value':updated['confidence_threshold'],'coverage':float(accepted.mean()),'accepted_accuracy':float((result.loc[accepted,'predicted']==result.loc[accepted,'class_index']).mean()) if accepted.any() else None}
    write_json(run/'reports/camera_test_metrics.json',report); result.to_csv(run/'reports/camera_test_predictions.csv',index=False)
    pd.DataFrame(report['report']).T.to_csv(run/'reports/classification_report.csv')
    cm=confusion_matrix(result.class_index,result.predicted,labels=range(4)); pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(run/'reports/confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(ax=ax,xticks_rotation=30); fig.tight_layout(); fig.savefig(run/'plots/confusion_matrix.png'); plt.close(fig)
    # Independent-process reload check without touching test data again.
    probe=np.zeros((1,224,224,3),np.float32); expected=chosen(probe,training=False).numpy(); np.save(run/'reports/reload_input.npy',probe)
    import subprocess
    code="import sys,numpy as np,tensorflow as tf; import src.utils; m=tf.keras.models.load_model(sys.argv[1],compile=False); np.save(sys.argv[3],m(np.load(sys.argv[2]),training=False).numpy())"
    subprocess.run([sys.executable,'-c',code,str(best_path),str(run/'reports/reload_input.npy'),str(run/'reports/reload_output.npy')],check=True)
    actual=np.load(run/'reports/reload_output.npy'); np.testing.assert_allclose(expected,actual,atol=1e-6,rtol=1e-5)
    write_json(run/'reports/reload_verification.json',{'passed':True,'max_absolute_difference':float(np.max(np.abs(expected-actual)))})
    for name in ['reload_input.npy','reload_output.npy']: (run/'reports'/name).unlink()
    write_json(run/'logs/run_summary.json',{'class_weights':weights,'baseline_validation_macro_f1':baseline_f1,'best_validation_macro_f1':best_f1,'epochs_completed':len(history.epoch),'camera_test_macro_f1':report['macro_f1'],'camera_test_accuracy':report['accuracy']})
    print(json.dumps(report,indent=2),flush=True); print('Saved independent camera run:',run)

if __name__=='__main__': main()
