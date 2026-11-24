"""Two-stage MobileNetV2 training; select checkpoint using validation only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import environment, dataset, build_model, write_json
from .config import CLASS_NAMES, CONFIDENCE_THRESHOLD, SEED

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default,kind in [('head-epochs',15,int),('fine-epochs',20,int),('batch',32,int),('head-lr',.001,float),('fine-lr',.00001,float),('fine-layers',25,int),('patience',4,int),('dropout',.3,float)]:
        p.add_argument('--'+name,default=default,type=kind)
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    if min(args.head_epochs,args.fine_epochs,args.batch,args.fine_layers,args.patience)<1:
        p.error('Epochs, batch, fine layers and patience must be positive')
    env=environment()
    write_json('outputs/logs/environment.json',env)
    for folder in ['models','outputs/logs','outputs/plots']: Path(folder).mkdir(parents=True,exist_ok=True)
    model,backbone=build_model(dropout=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.head_lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    if args.smoke:
        import numpy as np
        x=np.zeros((2,224,224,3),np.float32)
        model.train_on_batch(x,np.array([0,1])); y=model(x,training=False).numpy()
        assert y.shape==(2,4) and np.allclose(y.sum(1),1)
        print('Smoke test passed'); return
    frame=pd.read_csv('data/processed/manifest.csv')
    stats=json.loads(Path('data/processed/statistics.json').read_text())
    train=frame[frame.split=='train']; val=frame[frame.split=='val']
    weights={int(k):v for k,v in stats['class_weights'].items()}
    histories=[]; best=-1.; selected=None; batch=args.batch
    try:
        for stage,epochs,lr in [('head',args.head_epochs,args.head_lr),('fine',args.fine_epochs,args.fine_lr)]:
            if stage=='fine':
                model=tf.keras.models.load_model('models/waste_mobilenetv2.keras')
                backbone=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model) and 'mobilenet' in layer.name)
                backbone.trainable=True
                for layer in backbone.layers: layer.trainable=False
                for layer in backbone.layers[-args.fine_layers:]:
                    if not isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=True
            model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            initial=Path('models')/(stage+'_initial.keras'); model.save(initial)
            while True:
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',patience=args.patience,restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint('models/'+stage+'_best.keras',monitor='val_accuracy',mode='max',save_best_only=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2),
                    tf.keras.callbacks.CSVLogger('outputs/logs/'+stage+'.csv')]
                try:
                    hist=model.fit(dataset(train,batch,True),validation_data=dataset(val,batch),epochs=epochs,class_weight=weights,callbacks=callbacks,verbose=2)
                    break
                except tf.errors.ResourceExhaustedError:
                    if batch<=8: raise
                    batch=16 if batch>16 else 8
                    model=tf.keras.models.load_model(initial)
                    print('OOM: restart stage with batch',batch,flush=True)
            h=pd.DataFrame(hist.history); h['stage']=stage; h['epoch']=range(1,len(h)+1); histories.append(h)
            score=float(max(hist.history['val_accuracy']))
            if score>best:
                best=score; selected=stage
                chosen=tf.keras.models.load_model('models/'+stage+'_best.keras'); chosen.save('models/waste_mobilenetv2.keras')
            pd.concat(histories).to_csv('outputs/logs/history.csv',index=False)
            initial.unlink()
        history=pd.concat(histories,ignore_index=True)
        for metric in ['accuracy','loss']:
            fig,ax=plt.subplots(); ax.plot(history[metric],label='train'); ax.plot(history['val_'+metric],label='validation'); ax.set(xlabel='Epoch',ylabel=metric); ax.legend(); fig.savefig('outputs/plots/'+metric+'.png'); plt.close(fig)
        write_json('models/class_names.json',CLASS_NAMES)
        write_json('models/model_metadata.json',dict(architecture='MobileNetV2',input_size=[224,224,3],class_names=CLASS_NAMES,class_indices={c:i for i,c in enumerate(CLASS_NAMES)},label_mapping=stats['mapping'],preprocessing='RGB resize_with_pad 224x224, raw 0..255 input; model applies MobileNetV2 scaling exactly once',training_date=datetime.now(timezone.utc).isoformat(),seed=SEED,dataset_sources=sorted(frame.source.unique().tolist()),selected_checkpoint=selected,confidence_threshold=CONFIDENCE_THRESHOLD,tensorflow_version=tf.__version__,environment=env,training_config=vars(args),best_validation_accuracy=best,actual_batch=batch))
        print('Selected stage:',selected,'validation accuracy:',best,flush=True)
    except KeyboardInterrupt:
        model.save('models/interrupted.keras'); print('Interrupted state saved; evaluation not run'); raise SystemExit(130)

if __name__=='__main__': main()
