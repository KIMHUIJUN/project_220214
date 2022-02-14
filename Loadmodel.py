import keras.preprocessing.image
from keras.models import load_model
import sys
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image

from glob import glob
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
class_name = ['iu', 'suzy', 'uin']
newmodel = load_model('./model/02-1.0799.hdf5')

test_img = Image.open('./3.jpg').convert('L')
test_img = test_img.resize((128, 128))
test_img = np.array(test_img)
plt.imshow(test_img, cmap='Greys')
plt.show()

test_array = keras.preprocessing.image.img_to_array(test_img)
test_array = tf.expand_dims(test_img, 0)

predictions = newmodel.predict(test_array)
score = tf.nn.softmax(predictions[0])

print("{:.2f}perscent".format(100 * np.max(score)), class_name[np.argmax(score)])