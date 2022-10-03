import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import time
from termcolor import cprint

from tensorflow.keras import layers
from tensorflow.keras import losses
print(tf.__version__)

time.sleep (0.5)
cprint('=======', "blue")
print('IMDB Classification AI')
print('By Dirk & Niels')
cprint('=======', "blue")
time.sleep(1)

if not os.path.isdir('aclImdb'):
    cprint("IMDB Database not found, downloading form ai.stanford.edu...", "yellow")
    # Download dataset
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url,untar=True, cache_dir='.', cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
else:
    cprint('Dataset already present! Using ./aclImdb', "green")
    dataset_dir = './aclImdb'

train_dir = os.path.join(dataset_dir, 'train')

if os.path.isdir(os.path.join(train_dir, 'unsup')):
    cprint('Deleting unsup class reviews...', "yellow")
    shutil.rmtree(os.path.join(train_dir, 'unsup'))
    cprint('Done!', "green")

cprint('Creating training dataset...', "yellow")
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

cprint("Label 0 is", raw_train_ds.class_names[0], 'yellow')
cprint("Label 1 is", raw_train_ds.class_names[1]', yellow')

time.sleep(0.5)

cprint('Creating validation dataset...', "yellow")
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

cprint('Creating test dataset...', "yellow")
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)