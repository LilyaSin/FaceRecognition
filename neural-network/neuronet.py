
# coding: utf-8

# In[19]:

from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.utils import np_utils
from keras.models import Model, Sequential  

def create_model():
    height, width, deph = 96,96,1
    batch_size = 512
    num_epochs = 5
    hidden_size = 512 
    num_classes = 2 
    conv_depth_1 = 24
    pool_size = (2, 2)
    kernel_size = 2
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    inp = Input(shape=(1, height, width))
    conv_1 = Convolution2D(24, 2, 2, border_mode='same', activation='relu')(inp)
    drop_1 = Dropout(drop_prob_1)(conv_1)
    flat_1=Flatten()(drop_1)

    hidden_1 = Dense(batch_size, activation='relu')(flat_1)
    drop_2 = Dropout(drop_prob_2)(hidden_1)
    out = Dense(num_classes, activation='softmax')(drop_2)

    model = Model(input=inp, output=out)

    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model


# In[61]:

import os
import scipy.io as sio
import scipy

def read_jpeg_training_data():
    x_train = []
    y_train = []
    length = 0
    dirpath = 'data/positive'
    img_ext = '.png' 
    img_names = [ os.path.join(dirpath,x) for x in os.listdir( dirpath ) if x.endswith(img_ext) ]
    length += len(img_names)
    for i in img_names:
        x_train.append(read_jpeg(i))
        y_train.append(1)  
    dirpath = 'data/negative'
    img_names = [ os.path.join(dirpath,x) for x in os.listdir( dirpath ) if x.endswith(img_ext) ]
    for i in img_names:
        x_train.append(read_jpeg(i))
        y_train.append(0) 
    length += len(img_names)
    return x_train, y_train, length   

def train_model(model, x_test):
    batch_size = 512
    num_epochs = 5
    height, width, deph = 96,96,1
    x_train, y_train, set_size = read_jpeg_training_data()
    y_train = np_utils.to_categorical(y_train, 2) 
    x_train = np.array(x_train)
    x_train = x_train.astype('float32')
    x_train /= 225
    x_train = x_train.reshape(set_size,1,height,width)
    model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = num_epochs, verbose=0)
    predictions = model.predict(x_test, batch_size=32, verbose=0)
    return predictions


# Evaluator

# In[71]:

import numpy as np
import csv
from pandas import read_csv
from keras.utils import np_utils 
from PIL import Image

def evaluate_numpy_array(img):
    assert isinstance(img, np.ndarray)
    img = img.reshape(1,1,96,96)
    model = create_model()
    predictions = train_model(model, img)
    #print(predictions[0][1])
    return bool(int(round(predictions[0][1])))
        

def read_jpeg(path):
    im = Image.open(path).convert('L')
    X = list(im.getdata())
    X = np.array(X)
    return X
    
#img = read_jpeg('data/positive/outfile34.png') 
img = read_jpeg('data/negative/12.png') 
y_test = []
y_test.append(1)
y_test = np_utils.to_categorical(y_test, 2) 
print(evaluate_numpy_array(img))


# In[ ]:



