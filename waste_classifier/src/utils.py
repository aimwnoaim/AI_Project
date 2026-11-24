"""Shared preprocessing, streaming datasets and model construction."""
import json
import platform
from pathlib import Path
import numpy as np
import tensorflow as tf
from .config import CLASS_NAMES, SEED

@tf.keras.utils.register_keras_serializable(package='waste_classifier')
class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

def environment():
    tf.keras.utils.set_random_seed(SEED)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    info = dict(python=platform.python_version(), tensorflow=tf.__version__, devices=[str(g) for g in gpus] or ['CPU'])
    print(json.dumps(info), flush=True)
    return info

def preprocess(path):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return preprocess_rgb(image)

def preprocess_rgb(image):
    """Shared raw RGB preprocessing for files and live camera frames."""
    return tf.image.resize_with_pad(tf.cast(image, tf.float32), 224, 224)

def dataset(frame, batch=32, training=False, cache=False):
    paths = [str(Path(p).resolve()) for p in frame.path]
    ds = tf.data.Dataset.from_tensor_slices((paths, frame.class_index.astype('int32').values))
    ds = ds.map(lambda p,y: (preprocess(p),y), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache(str(Path('data/processed/cache_' + ('train' if training else 'eval')).resolve()))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def build_model(weights='imagenet', dropout=.3):
    backbone = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=weights)
    backbone.trainable = False
    inputs = tf.keras.Input((224,224,3))
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal', seed=SEED),
        tf.keras.layers.RandomRotation(.04, seed=SEED+1),
        tf.keras.layers.RandomZoom(.1, seed=SEED+2),
        tf.keras.layers.RandomBrightness(.15, value_range=(0,255), seed=SEED+3),
        tf.keras.layers.RandomContrast(.15, seed=SEED+4)], name='augmentation')
    x = augment(inputs)
    x = MobileNetPreprocess(name='mobilenet_preprocess')(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x))
    return model, backbone

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')
