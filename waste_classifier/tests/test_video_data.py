"""Synthetic video import/splitting tests; no camera or real training."""
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import pandas as pd
from src.config import CLASS_NAMES
from src.video_data import extract, identity, prepare

class VideoDataTests(unittest.TestCase):
    def test_finetune_rejects_item_leakage(self):
        from src.finetune_camera import validate_manifest
        rows=[dict(path='unused',label=label,class_index=i,split=split,item_id=f'{label}_{split}',group=f'{label}_{split}',sha256=f'{label}_{split}',source='camera_video') for i,label in enumerate(CLASS_NAMES) for split in ['train','val','test']]
        frame=pd.DataFrame(rows); validate_manifest(frame)
        frame.loc[1,'item_id']=frame.loc[0,'item_id']
        with self.assertRaises(ValueError): validate_manifest(frame)

    def test_identity(self):
        self.assertEqual(identity('plastic__bottle01__front.mp4'),('plastic','bottle01'))
        self.assertEqual(identity('unnamed.mp4','metal','can01'),('metal','can01'))
        with self.assertRaises(ValueError): identity('unnamed.mp4')
        with self.assertRaises(ValueError): identity('glass__object.mp4')

    def test_import_and_grouped_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); inventory=root/'inventory'
            for c,label in enumerate(CLASS_NAMES):
                for item in range(4):
                    path=root/f'{label}__{label}{item}__clip.avi'
                    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),4,(32,24))
                    self.assertTrue(writer.isOpened())
                    for i in range(8):
                        image=np.random.default_rng(c*100+item*10+i).integers(0,255,(24,32,3),dtype=np.uint8)
                        writer.write(image)
                    writer.release()
                    rows=extract(path,inventory,every_seconds=1,max_frames=2)
                    self.assertEqual(len(rows),2)
            with self.assertRaises(ValueError): extract(path,inventory)
            frame=prepare(inventory/'frames.csv',root/'split')
            self.assertEqual(frame.groupby('item_id').split.nunique().max(),1)
            self.assertEqual(frame.groupby('session_id').split.nunique().max(),1)
            self.assertEqual(frame.groupby('sha256').split.nunique().max(),1)
            self.assertEqual(set(frame.split),{'train','val','test'})
            other=prepare(inventory/'frames.csv',root/'repeat')
            self.assertEqual(frame.split.tolist(),other.split.tolist())
            with self.assertRaises(FileExistsError): prepare(inventory/'frames.csv',root/'split')

    def test_too_few_items(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest=Path(temp)/'empty.csv'
            pd.DataFrame(columns=['path']).to_csv(manifest,index=False)
            with self.assertRaises(ValueError): prepare(manifest,Path(temp)/'split')

if __name__=='__main__': unittest.main()
