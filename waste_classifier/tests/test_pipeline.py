"""Synthetic-only pipeline checks; never trains on evaluation data."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from src.config import CLASS_NAMES,map_label
from src.download_dataset import safe_extract
from src.prepare_dataset import discover,split_rows
from src.predict import format_prediction
from src.utils import build_model,preprocess

class PipelineTests(unittest.TestCase):
    def test_class_order_and_mapping(self):
        self.assertEqual(CLASS_NAMES,['plastic','metal','paper_cardboard','other'])
        for category,label in [('Plastic','plastic'),('plastic','plastic'),('Metal','metal'),('metal','metal'),('Paper','paper_cardboard'),('cardboard','paper_cardboard'),('Glass','other'),('Food Organics','other'),('unknown','other')]:
            self.assertEqual(map_label(category),label)

    def test_zip_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError): safe_extract(archive,Path(temp)/'out')

    def test_discovery_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            for category in ['plastic','metal','paper','cardboard','glass','trash']:
                folder=Path(temp)/category; folder.mkdir()
                Image.new('RGB',(20,10),(10,20,30)).save(folder/'image.jpg')
            (Path(temp)/'plastic'/'bad.jpg').write_bytes(b'not an image')
            rows,errors=discover(temp,'trashnet')
            self.assertEqual(len(rows),6); self.assertEqual(len(errors),1)

    def test_split_determinism_and_duplicate_groups(self):
        rows=[dict(path=f'{c}/{i}',class_index=c,sha256=f'{c}-{i}',item_id='',session_id='') for c in range(4) for i in range(80)]
        rows.append(dict(rows[0],path='duplicate'))
        rows[1]['item_id']=rows[2]['item_id']='same-item'
        rows[2]['session_id']=rows[3]['session_id']='same-session'
        a=split_rows(pd.DataFrame(rows)); b=split_rows(pd.DataFrame(rows))
        self.assertEqual(a.split.tolist(),b.split.tolist())
        self.assertEqual(a.groupby('sha256').split.nunique().max(),1)
        self.assertEqual(a.loc[[1,2,3],'split'].nunique(),1)
        self.assertEqual(set(a.split),{'train','val','test'})

    def test_threshold_and_json(self):
        result=format_prediction([.92,.03,.03,.02])
        self.assertTrue(result['should_sort']); self.assertEqual(result['label'],'plastic')
        self.assertEqual(set(json.loads(json.dumps(result))),{'label','confidence','should_sort','probabilities'})
        retry=format_prediction([.4,.3,.2,.1]); self.assertEqual(retry['label'],'retry'); self.assertAlmostEqual(retry['confidence'],.4)
        self.assertEqual(format_prediction([.01,.01,.01,.97])['label'],'other')
        with self.assertRaises(ValueError): format_prediction([0,0,0,0])

    def test_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'image.png'; Image.new('RGB',(80,20),'white').save(path)
            x=preprocess(str(path)); self.assertEqual(tuple(x.shape),(224,224,3))
            self.assertEqual(float(tf.reduce_max(x)),255.)
            self.assertEqual(float(tf.reduce_max(x[0])),0.)

    def test_model_save_load(self):
        model,_=build_model(weights=None)
        x=np.ones((1,224,224,3),np.float32)*100
        y=model(x,training=False).numpy(); self.assertEqual(y.shape,(1,4)); np.testing.assert_allclose(y.sum(1),1,atol=1e-6)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.keras'; model.save(path)
            restored=tf.keras.models.load_model(path,compile=False)
            np.testing.assert_allclose(y,restored(x,training=False).numpy(),atol=1e-6)

if __name__=='__main__': unittest.main()
