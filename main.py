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

cprint("Label 0 is " + str(raw_train_ds.class_names[0]), 'yellow')
cprint("Label 1 is " + str(raw_train_ds.class_names[1]), 'yellow')

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

def cleanup(input_data):
  stripped_html = tf.strings.regex_replace(tf.strings.lower(input_data), '<br />', ' ')
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation),'')

# begin model

vectorize_layer = layers.TextVectorization(
    standardize=cleanup,
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250)

# adapt vectorize layer 
vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))


# tijddelijk
# def vectorize_text(text, label):
#   text = tf.expand_dims(text, -1)
#   return vectorize_layer(text), label

# text_batch, label_batch = next(iter(raw_train_ds))
# fr, fl = text_batch[0], label_batch[0]
# print("Review:\n    ", cleanup(fr))
# print("Label:\n", raw_train_ds.class_names[fl])
# print("Vectorized:\n", vectorize_text(fr, fl))

for x in range (0, 50):
    print(str(x) + " ---> ", vectorize_layer.get_vocabulary()[x])