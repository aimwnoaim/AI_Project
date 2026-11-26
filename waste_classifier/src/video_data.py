"""Import labeled local videos and prepare item-grouped camera splits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from .config import CLASS_NAMES, SEED

FIELDS=['path','label','class_index','source','original_category','item_id','session_id','video_sha256','frame_index','timestamp_seconds','sha256']

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def identity(video,label=None,item_id=None):
    parts=Path(video).stem.split('__')
    label=label or (parts[0] if len(parts)>=2 else None)
    item_id=item_id or (parts[1] if len(parts)>=2 else None)
    if label not in CLASS_NAMES: raise ValueError('Provide --label or name video CLASS__ITEM__CLIP.mp4')
    if not item_id or not re.fullmatch(r'[A-Za-z0-9_-]+',item_id):
        raise ValueError('Provide an item ID using letters, digits, underscores or hyphens')
    return label,item_id

def extract(video,output,label=None,item_id=None,every_seconds=1.,max_frames=150):
    video=Path(video).resolve(); output=Path(output)
    if not video.is_file(): raise FileNotFoundError(video)
    label,item_id=identity(video,label,item_id)
    if every_seconds<=0 or max_frames<1: raise ValueError('Interval and max frames must be positive')
    digest=file_hash(video)
    manifest=output/'frames.csv'
    old=pd.read_csv(manifest,dtype={'item_id':str}) if manifest.exists() else pd.DataFrame(columns=FIELDS)
    if digest in set(old.video_sha256): raise ValueError('This video is already imported')
    if not old.empty and any(old.loc[old.item_id==item_id,'label']!=label): raise ValueError('Item ID already has a different label')
    capture=cv2.VideoCapture(str(video)); created=[]; rows=[]
    try:
        if not capture.isOpened(): raise ValueError(f'Cannot decode video: {video}')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0: raise ValueError('Video has no usable FPS metadata; transcode to MP4 first')
        step=max(1,round(fps*every_seconds)); index=0
        folder=output/'frames'/label; folder.mkdir(parents=True,exist_ok=True)
        while len(rows)<max_frames:
            ok,frame=capture.read()
            if not ok: break
            if index%step==0:
                path=folder/f'{item_id}_{digest[:12]}_{index:08d}_{uuid.uuid4().hex[:8]}.jpg'
                if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,95]): raise OSError(f'Cannot write {path}')
                created.append(path)
                rows.append(dict(path=path.resolve().as_posix(),label=label,class_index=CLASS_NAMES.index(label),source='camera_video',original_category=label,item_id=item_id,session_id=digest,video_sha256=digest,frame_index=index,timestamp_seconds=index/fps,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            index+=1
        if not rows: raise ValueError('Video yielded no readable frames')
        output.mkdir(parents=True,exist_ok=True)
        combined=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
        temporary=manifest.with_suffix('.tmp'); combined.to_csv(temporary,index=False); temporary.replace(manifest)
    except Exception:
        for path in created: path.unlink(missing_ok=True)
        raise
    finally: capture.release()
    print(f'Imported {len(rows)} frames: {label}, item={item_id}, manifest={manifest}',flush=True)
    return rows

def prepare(manifest,output):
    frame=pd.read_csv(manifest,dtype={'item_id':str})
    if not set(FIELDS).issubset(frame): raise ValueError('Invalid frame manifest')
    for row in frame.itertuples():
        if row.label not in CLASS_NAMES or row.class_index!=CLASS_NAMES.index(row.label): raise ValueError('Invalid class order/label')
        with Image.open(row.path) as image: image.verify()
        if hashlib.sha256(Path(row.path).read_bytes()).hexdigest()!=row.sha256: raise ValueError(f'Frame modified: {row.path}')
    if frame.groupby('item_id').label.nunique().max()>1: raise ValueError('Item ID has conflicting labels')
    if frame.groupby('sha256').label.nunique().max()>1: raise ValueError('Duplicate image has conflicting labels')
    # Union item IDs linked by duplicate frame bytes; these groups cannot cross splits.
    parent={item:item for item in frame.item_id.unique()}
    def find(item):
        while parent[item]!=item: parent[item]=parent[parent[item]]; item=parent[item]
        return item
    for _,group in frame.groupby('sha256'):
        items=group.item_id.tolist()
        for item in items[1:]: parent[find(item)]=find(items[0])
    frame['group']=frame.item_id.map(find)
    frame=frame.drop_duplicates('sha256').copy()
    rng=np.random.default_rng(SEED); assignments={}
    for label in CLASS_NAMES:
        groups=sorted(frame.loc[frame.label==label,'group'].unique())
        if len(groups)<3: raise ValueError(f'{label}: need at least 3 independent item groups (10+ recommended); found {len(groups)}')
        rng.shuffle(groups); n=max(1,round(len(groups)*.15)); n=min(n,(len(groups)-1)//2)
        for i,group in enumerate(groups): assignments[group]='test' if i<n else ('val' if i<2*n else 'train')
    frame['split']=frame.group.map(assignments)
    out=Path(output)
    if out.exists(): raise FileExistsError(f'Output already exists: {out}; choose a new split directory')
    out.mkdir(parents=True)
    frame.to_csv(out/'manifest.csv',index=False)
    stats={'seed':SEED,'counts':frame.groupby(['split','label']).size().reset_index(name='count').to_dict('records'),'item_counts':frame.groupby(['split','label']).item_id.nunique().reset_index(name='items').to_dict('records'),'split_method':'Approximately 70/15/15 of independent item groups per class; frame ratios vary with video length.'}
    (out/'statistics.json').write_text(json.dumps(stats,indent=2))
    print(frame.groupby(['split','label']).size().to_string(),flush=True)
    return frame

def main():
    p=argparse.ArgumentParser(description=__doc__); commands=p.add_subparsers(dest='command',required=True)
    e=commands.add_parser('extract'); e.add_argument('--video',type=Path); e.add_argument('--video-dir',type=Path); e.add_argument('--label',choices=CLASS_NAMES); e.add_argument('--item-id'); e.add_argument('--output',default='data/camera'); e.add_argument('--every-seconds',type=float,default=1); e.add_argument('--max-frames',type=int,default=150)
    s=commands.add_parser('prepare'); s.add_argument('--manifest',default='data/camera/frames.csv'); s.add_argument('--output',default='data/camera/split_v1')
    a=p.parse_args()
    try:
        if a.command=='prepare': prepare(a.manifest,a.output)
        else:
            if bool(a.video)==bool(a.video_dir): raise ValueError('Choose exactly one of --video or --video-dir')
            videos=[a.video] if a.video else sorted(v for v in a.video_dir.iterdir() if v.suffix.lower() in {'.mp4','.mov','.avi','.mkv','.webm'})
            if not videos: raise ValueError('No videos found')
            for video in videos: extract(video,a.output,a.label,a.item_id,a.every_seconds,a.max_frames)
    except (ValueError,OSError) as exc: p.exit(1,f'Video data error: {exc}\n')

if __name__=='__main__': main()
