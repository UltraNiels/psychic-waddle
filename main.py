import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import time

from tensorflow.keras import layers
from tensorflow.keras import losses
print(tf.__version__)

time.sleep (0.5)
print('=======')
print('IMDB Classification AI')
print('By Dirk & Niels')
print('=======')
time.sleep(1)

if not os.path.isdir('aclImdb'):
    print("IMDB Database not found, downloading form ai.stanford.edu...")
    # Download dataset
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url,untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
else:
    print('Dataset already present! Using ./aclImdb')
    dataset_dir = './aclImdb'

train_dir = os.path.join(dataset_dir, 'train')

if os.path.isdir(os.path.join(train_dir, 'unsup')):
    print('Deleting unsup class reviews...')
    shutil.rmtree(os.path.join(train_dir, 'unsup'))
    print('Done!')

print('Creating validation set...')
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)