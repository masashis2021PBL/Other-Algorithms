import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten

import tensorflow.keras.callbacks as callbacks
import tensorflow as tf

import os
import copy
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import plotly
import seaborn as sns

plotly.offline.init_notebook_mode (connected=False)
warnings.filterwarnings ('ignore')
sns.set_style ("whitegrid", {'grid.linestyle': '--'})
dfstyle = [dict (selector = "th", props = [('font-size', '14px')]), dict (selector="td", props = [('font-size', '16px')])]

SEED = 800
tf.random.set_seed (SEED)
np.random.seed (SEED)

def sin (x, T=300):
    return np.sin (2.0 * np.pi * x / T)

def toy_problem (T = 100, amp = 0.05):
    x = np.arange (0, 2 * T + 1)
    return sin (x)

def make_dataset (raw_data, n_prev = 100, maxlen = 25):
    data, target  = [], []
   
    for i in range (len (raw_data) - maxlen):
        data.append (raw_data [i:(i + maxlen)])
        target.append (raw_data [i + maxlen])

    reshaped_data = np.array (data).reshape (len (data), maxlen, 1)
    reshaped_target = np.array (target).reshape (len (target), 1)

    return reshaped_data, reshaped_target

function = toy_problem (T = 300)

data, label = make_dataset (function, maxlen = 50)
print (data.shape)

inputs = Input(shape=(50, 1))


model = models.Sequential ()
model.add (Conv1D (30, 2, padding = 'same'))
model.add (Activation ('relu'))
model.add (MaxPool1D (pool_size = 2, padding = 'same'))
model.add (Conv1D (10, 2, padding = 'same'))
model.add (Activation ('tanh'))
model.add (MaxPool1D (pool_size = 2, padding = 'same'))
model.add (Conv1D (10, 2, padding = 'same'))
model.add (Activation ('tanh'))
model.add (MaxPool1D (pool_size = 2, padding = 'same'))
model.add (Flatten ())
model.add (Dense (300))
model.add (Activation ('relu'))
model.add (Dense (1))
model.add (Activation ('tanh'))

#model = Model (inputs, outputs=x)

model.compile (loss = "mean_squared_error", optimizer = Adam(lr = 1e-3),
              metrics = ['accuracy'])

# Early Stopping
early_stopping = callbacks.EarlyStopping (monitor = 'val_loss', mode = 'min', patience = 10)

model.fit (data, label,
          batch_size = 64, epochs = 500,
          validation_split = 0.2, callbacks = [early_stopping]
         )

predicted  = model.predict (data)

future_test = data [-1].T
time_length = future_test.shape [1]
future_result = np.empty ( (0) )

for step in range(400):
    test_data = np.reshape (future_test, (1, time_length, 1))
    batch_predict = model.predict (test_data)

    future_test = np.delete (future_test, 0)
    future_test = np.append (future_test, batch_predict)

    future_result = np.append(future_result, batch_predict)

fig = plt.figure (figsize = (16, 9), dpi=200)

sns.lineplot (
    color = "#fe90af",
    data = function,
    label = "Raw Data"
)

sns.lineplot (
    color = "#61c0bf",
    x = np.arange (50, len (predicted) + 50),
    y = predicted.reshape (-1),
    label = "Predicted Training Data"
)

sns.lineplot (
    color = "#81c00e",
    x = np.arange (0 + len (function), len (function) + len (future_result)),
    y = future_result.reshape (-1),
    label = "Predicted Future Data"
)
