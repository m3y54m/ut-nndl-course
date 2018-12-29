from __future__ import print_function
from keras.models import Sequential, load_model
import keras
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Flatten,RNN, SimpleRNN
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import keras.backend as K


'''
IN this part data for test, validation and train is preperd.
'''
# data format is : pollution  ,dew  ,temp  , press ,wnd_dir , wnd_spd , snow , rain
# total data =43000
# per day and week = 43000/24*7 =250
count = 24*7                        # if 1 all data, if 24 per day, if 24*7 per day & week
train_c = 120
val_c = 50
test_c = 30
data = np.load('polution_dataSet.npy')
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

week = 8            # sample count for prediction
hour = 2            # which hour data ? (0-23) (not important)

for i in range(train_c):
    x_train.append(np.array(data[i*count+hour:i*count+week+hour]))
    y_train.append(np.array(data[i*count+week+hour, 0]))
x_train = np.array(x_train)
y_train = np.array(y_train)

for i in range(train_c, train_c+val_c):
    x_val.append(np.array(data[i*count+hour:i*count+week+hour]))
    y_val.append(np.array(data[i*count+week+hour, 0]))
x_val = np.array(x_val)
y_val = np.array(y_val)

for i in range(train_c+val_c, train_c+val_c+test_c):
    x_test.append(np.array(data[i*count+hour:i*count+week+hour]))
    y_test.append(np.array(data[i*count+week+hour, 0]))
x_test = np.array(x_test)
y_test = np.array(y_test)
y_total = np.concatenate([y_train, y_val, y_test], axis=None)


'''
In this part network is implemented 
'''

# parameters
batch_size = 20
n_epochs = 25
hidden_size = 4


# class for RNN Keras
class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


# Let's use this cell in a RNN layer:
cell = MinimalRNNCell(40)

# keras modeling
model = Sequential()
model.add(RNN(cell,  input_shape=(week, 8), return_sequences=False))
# model.add(RNN(cell,  input_shape=(24, 8), return_sequences=False))
# model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
print(model.summary())
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(x_val, y_val))

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

print('train loss history:', history.history["loss"])
print('test loss history:', history.history["val_loss"])

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print('\nTest result loss: %.3f' % (scores))

x_old = np.concatenate((x_train, x_val), axis=0)
y_pred = model.predict(x_test, batch_size=batch_size)
y_old = model.predict(x_old, batch_size=batch_size)

xc = range(n_epochs)
plt.figure(1, figsize=(7, 5))
plt.subplot(211)
plt.plot(range(0, train_c+val_c), y_old, c='r', linewidth=3.0)
plt.plot(range(train_c+val_c, train_c+val_c+test_c), y_pred, c='b', linewidth=3.0)
plt.plot(range(0, train_c+val_c+test_c), y_total, 'c')
plt.xlabel('data')
plt.ylabel('value')
plt.title('LSTM ')
plt.grid(True)
plt.legend(['prediction for train', 'prediction for test',' exact'])
plt.style.use(['classic'])


plt.subplot(212)
plt.plot(range(train_c+val_c, train_c+val_c+test_c), y_pred, 'b', linewidth=3.0)
plt.plot(range(train_c+val_c, train_c+val_c+test_c), y_total[train_c+val_c:train_c+val_c+test_c], 'c')
plt.xlabel('data')
plt.ylabel('value')
plt.title('value')
plt.grid(True)
plt.legend(['prediction for test',' exact'])
plt.style.use(['classic'])
plt.show()