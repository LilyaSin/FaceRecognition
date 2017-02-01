
# coding: utf-8

# In[42]:

from keras.datasets import mnist #извлечение данных МНИСТа 
FTRAIN = 'data/training.csv' #getting training data


# In[53]:

from lasagne import layers
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import theano


# In[35]:

from keras.models import Model  #класс для определения и обучений НС


# In[72]:

import os
from pandas import DataFrame
from pandas.io.parsers import read_csv


# In[36]:

from keras.layers import Input, Dense # 2 типа слоев НС


# Dense-слой это обычный линейный юнит, который взвешенно суммирует компоненты входного вектора.

# In[37]:

from keras.utils import np_utils # утилита для декодирования истенных значений (перевод в Унитарный/Двоичный код)


# In[62]:

import numpy as np 
from sklearn.base import clone


# In[51]:

def float32(k):
    return np.cast['float32'](k)


# In[39]:

batch_size = 64 # рассматриваем 128 обучающий парметров на каждой итерации
num_epochs = 1 # итерируемся 20 раз по обучающей выборке 
hidden_size = 128 # по 512 нейронов в каждом (из двух) слоев НС


# batch_size — количество обучающих образцов, обрабатываемых одновременно за одну итерацию алгоритма градиентного спуска
# 
# num_epochs — количество итераций обучающего алгоритма по всему обучающему множеству;
# 
# hidden_size — количество нейронов в каждом из двух скрытых слоев MLP.

# In[77]:

""""num_train = 60000 # 60к тренеровочных МНИСТ примеров
num_test = 10000 #10к тестовых МНИСТ примеров

height, width, deph = 28,28,1 #параметры изображение 28x28 в сером спектре
num_classes = 10 # 10 классов 

(X_train, y_train), (X_test, y_test) = mnist.load_data() # загрузка выборки мнист

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#нормализуем даные в диапазоне [0;1]
X_train /= 225
X_test /= 225

#кодирование меток в двоичный код
Y_train = np_utils.to_categorical(y_train, num_classes) 
Y_test = np_utils.to_categorical(y_test,num_classes) """""

def load():
    cols = None
    fname = FTRAIN
    df = read_csv(os.path.expanduser(fname))  

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df[list(cols) + ['Image']]

    print(df.count())  
    df = df.dropna()  

    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)
    y = None

    return X, y


# In[55]:

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb


# In[57]:

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


# In[64]:

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


# Определение модели

# In[65]:

#inp = Input(shape=(height*width,)) #на входе вектор размера 784
#hidden_1 = Dense(hidden_size, activation ='relu')(inp) #первый скрытый ReLu слой
#hidden_2 = Dense(hidden_size, activation = 'relu')(hidden_1) #второй скрытый ReLu слой
#out = Dense(num_classes, activation='softmax')(hidden_2) # выходной softmax слой
# softmax превращает вектор действительных чисел в вектор вероятностей

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    hidden1_num_units=1000,
    hidden2_num_units=1000,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    max_epochs=1,
    verbose=1,
    )

#model = Model(input = inp, output = out)
model = clone(net)


# Определим функцю потерь (оптимизатор Адама)

# In[11]:

model.compile(loss='categorical_crossentropy', #функция потерь крос - энтропии
             optimizer = 'adam', #отпимизатор Адама
              metrics = ['accuracy'] #точность - доля входных данных, отнесенных к правильному классу
             )


# In[66]:

#model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = num_epochs, verbose = 1,validation_split = 0.1) #оставляем 10% данных для проверки
#model.evaluate(X_test, Y_test,verbose=1) #оценка модели на тестовом наборе 

#print("Training model for columns {} for {} epochs".format(cols, model.max_epochs))
model.fit(X_train, Y_train)


# In[78]:

load()


# In[ ]:



