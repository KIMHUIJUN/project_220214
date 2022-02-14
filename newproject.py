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

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

image_datas = glob('./img/*/*.jpg')
class_name = ['iu', 'suzy', 'uin']
dic = {"iu": 0, 'suzy': 1, 'uin': 2}

X = [] # 이미지 저장 리스트
Y = [] #라벨 저장 리스트

for imagename in image_datas:
    image = Image.open(imagename).convert('L') # 이미지 개별로 불러오기
    image = image.resize((128, 128)) #이미지 사이즈 조절 (128, 128)
    image = np.array(image) #이미지 수열로 변화
    X.append(image) # X리스트에 이미지 넣기
    label = imagename.split('\\')[1]
    Y.append(dic[label]) # 사진별 라벨 Y 리스트에 넣기

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_class_train, Y_class_test = train_test_split(X, Y, test_size=0.2, shuffle= True, random_state= 44)

print(len(X))
print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))
print('테스트셋 이미지 수: %d 개' % (X_test.shape[0]))

plt.imshow(X_train[0], cmap='Greys')
plt.show()



X_train = X_train.reshape(X_train.shape[0], 128, 128,1)

X_train = X_train.astype('float32')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 128, 128,1).astype('float32') / 255

print('Class : %d' % (Y_class_train[0]))


Y_train = np_utils.to_categorical(Y_class_train)
Y_test = np_utils.to_categorical(Y_class_test)

print(Y_train[0])
print(X_train.shape)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), input_shape=(128, 128, 1),activation='relu'))
model.add(Conv2D(64,(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

Model_DIR = './model/'
if not os.path.exists(Model_DIR):
    os.mkdir(Model_DIR)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=10)

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=30, batch_size=20, verbose=0, callbacks=[early_stopping_callback, checkpointer])
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))

y_vloss = history.history['val_loss']

y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()