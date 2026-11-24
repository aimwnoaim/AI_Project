"""Validate, deduplicate and create deterministic grouped stratified manifests."""
import argparse
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from .config import CLASS_NAMES, SEED, map_label
from .download_dataset import safe_extract
import json

def discover(path, source):
    path = Path(path)
    if path.is_file():
        path = safe_extract(path, Path('data/raw') / source)
    expected = {'plastic','metal','paper','cardboard','glass','trash'} if source=='trashnet' else {'Plastic','Metal','Paper','Cardboard','Glass','Food Organics','Miscellaneous Trash','Textile Trash','Vegetation'}
    folders = {p.name for p in path.rglob('*') if p.is_dir()}
    if not expected.issubset(folders):
        raise ValueError(f'{source}: missing categories {expected-folders}')
    rows, rejected = [], []
    for image in sorted(path.rglob('*')):
        if image.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'} or '__MACOSX' in image.parts:
            continue
        try:
            with Image.open(image) as im:
                im.verify()
            with Image.open(image) as im:
                im.convert('RGB').load()
            category = image.parent.name
            label = map_label(category)
            rows.append(dict(path=image.as_posix(), source=source, original_category=category, label=label,
                             class_index=CLASS_NAMES.index(label), sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
        except Exception as exc:
            rejected.append(dict(path=str(image), error=str(exc)))
    if not rows:
        raise ValueError(f'{source}: no readable images')
    return rows, rejected

def split_rows(frame, metadata=None):
    frame = frame.copy().reset_index(drop=True)
    parent = list(range(len(frame)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen = {}
    for i,row in frame.iterrows():
        keys = [('hash',row.sha256)]
        for field in ['item_id','session_id']:
            value = row.get(field,'')
            if pd.notna(value) and str(value):
                keys.append((field,str(value)))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key]=i
    frame['group']=[find(i) for i in range(len(frame))]
    # Twenty stratified folds yield 14/3/3 folds = approximately 70/15/15.
    splitter=StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    frame['split']='train'
    for fold,(_,indices) in enumerate(splitter.split(frame,frame.class_index,frame.group)):
        frame.loc[indices,'split']='test' if fold<3 else ('val' if fold<6 else 'train')
    return frame

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--realwaste-path',default='data/raw/realwaste')
    p.add_argument('--trashnet-path',default=None,help='Optional second dataset; omitted for RealWaste-only runs')
    p.add_argument('--metadata',type=Path)
    args=p.parse_args()
    rows=[]; rejected=[]
    for source in ['realwaste','trashnet']:
        if getattr(args,source+'_path') is None: continue
        r,e=discover(getattr(args,source+'_path'),source); rows+=r; rejected+=e
    frame=pd.DataFrame(rows)
    if args.metadata:
        frame=frame.merge(pd.read_csv(args.metadata).fillna(''),on='path',how='left',validate='one_to_one')
    conflicts=frame.groupby('sha256').class_index.nunique()
    bad=set(conflicts[conflicts>1].index)
    if bad:
        rejected += frame[frame.sha256.isin(bad)].to_dict('records')
        frame=frame[~frame.sha256.isin(bad)]
    frame=split_rows(frame)
    out=Path('data/processed'); out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'manifest.csv',index=False)
    mapping=frame[['source','original_category','label']].drop_duplicates().sort_values(['source','original_category'])
    print(mapping.to_string(index=False),flush=True)
    train=frame[frame.split=='train']; counts=train.class_index.value_counts()
    stats={'counts':frame.groupby(['source','split','label','original_category']).size().reset_index(name='count').to_dict('records'),
           'class_weights':{str(i):len(train)/(4*int(counts[i])) for i in range(4)},
           'duplicate_rows':int(frame.sha256.duplicated().sum()),'rejected':rejected,'mapping':mapping.to_dict('records'),
           'limitation':'No physical-item IDs are provided. Item independence cannot be guaranteed; this test does not prove performance on unseen physical objects.'}
    (out/'statistics.json').write_text(json.dumps(stats,indent=2))
    print(frame.groupby(['source','split','label']).size().to_string(),flush=True)
    custom=[]
    for label in CLASS_NAMES:
        folder=Path('data/custom')/label
        for image in sorted(folder.glob('*')):
            if image.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
            try:
                with Image.open(image) as im: im.verify()
                digest=hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in set(frame.sha256): raise ValueError('Custom image overlaps main dataset')
                custom.append(dict(path=image.as_posix(),source='custom',original_category=label,label=label,class_index=CLASS_NAMES.index(label),sha256=digest,split='custom_test'))
            except Exception as exc: raise ValueError(f'Invalid custom image {image}: {exc}')
    if custom: pd.DataFrame(custom).to_csv(out/'custom_manifest.csv',index=False)

if __name__=='__main__': main()
