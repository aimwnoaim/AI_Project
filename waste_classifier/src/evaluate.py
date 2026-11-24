"""Evaluate the selected model on untouched test and optional custom sets."""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import dataset,write_json
from .config import CLASS_NAMES

def metrics(frame):
    y=frame.class_index; pred=frame.predicted
    return dict(count=len(frame),accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=range(4),average='macro',zero_division=0),weighted_f1=f1_score(y,pred,labels=range(4),average='weighted',zero_division=0),report=classification_report(y,pred,labels=range(4),target_names=CLASS_NAMES,output_dict=True,zero_division=0))

def main():
    out=Path('outputs/reports'); out.mkdir(parents=True,exist_ok=True)
    if (out/'test_metrics.json').exists(): raise SystemExit('Test report already exists; refusing to silently repeat test evaluation')
    model=tf.keras.models.load_model('models/waste_mobilenetv2.keras',compile=False)
    sets=[('test',pd.read_csv('data/processed/manifest.csv').query("split == 'test'"))]
    if Path('data/processed/custom_manifest.csv').exists(): sets.append(('custom',pd.read_csv('data/processed/custom_manifest.csv')))
    for name,frame in sets:
        probabilities=model.predict(dataset(frame),verbose=0); frame=frame.copy(); frame['predicted']=probabilities.argmax(1); frame['confidence']=probabilities.max(1)
        for i,label in enumerate(CLASS_NAMES): frame['prob_'+label]=probabilities[:,i]
        report=metrics(frame); report['per_source']={s:metrics(group) for s,group in frame.groupby('source')}
        pd.DataFrame([dict(source=source,**{k:v for k,v in scores.items() if k!='report'}) for source,scores in report['per_source'].items()]).to_csv(out/(name+'_source_metrics.csv'),index=False)
        report['other_original_categories']={}
        for (source,category),group in frame[frame.label=='other'].groupby(['source','original_category']):
            report['other_original_categories'][source+'/'+category]={'count':len(group),'predicted_counts':{CLASS_NAMES[int(k)]:int(v) for k,v in group.predicted.value_counts().items()}}
        accepted=frame.confidence>=.8
        report['threshold_0.8']={'coverage':float(accepted.mean()),'accuracy_accepted':float((frame.loc[accepted,'predicted']==frame.loc[accepted,'class_index']).mean()) if accepted.any() else None}
        write_json(out/(name+'_metrics.json'),report)
        frame.to_csv(out/(name+'_predictions.csv'),index=False)
        pd.DataFrame(report['report']).T.to_csv(out/(name+'_classification_report.csv'))
        cm=confusion_matrix(frame.class_index,frame.predicted,labels=range(4))
        pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(out/(name+'_confusion_matrix.csv'))
        fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(ax=ax,xticks_rotation=30); fig.tight_layout(); fig.savefig('outputs/plots/'+name+'_confusion_matrix.png'); plt.close(fig)
        print(name,report,flush=True)

if __name__=='__main__': main()
