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
print('IMDB NN')
print('=======')
time.sleep(1)

dataset_dir = './aclImdb'
train_dir = os.path.join(dataset_dir, 'train')

print('Creating validation set...')
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)