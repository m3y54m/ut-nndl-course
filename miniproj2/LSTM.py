from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Flatten
from keras.layers import LSTM
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

'''
IN this part data for test, validation and train is preperd.
'''
# data format is : pollution  ,dew  ,temp  , press ,wnd_dir , wnd_spd , snow , rain
# total data =43000
# per day and week = 43000/24*7 =250
SAMPLE_RATE = 1                       # if 1 all data, if 24 per day, if 24*7 per day & week
TRAIN_SIZE = 7000
VAL_SIZE = 1000
TEST_SIZE = 2000
data = np.load('polution_dataSet.npy')
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

TIMESTEPS = 24            # sample count for prediction

for i in range(TRAIN_SIZE):
    x_train.append(np.array(data[i*SAMPLE_RATE:i*SAMPLE_RATE+TIMESTEPS]))
    y_train.append(np.array(data[i*SAMPLE_RATE+TIMESTEPS, 0]))
x_train = np.array(x_train)
y_train = np.array(y_train)

for i in range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE):
    x_val.append(np.array(data[i*SAMPLE_RATE:i*SAMPLE_RATE+TIMESTEPS]))
    y_val.append(np.array(data[i*SAMPLE_RATE+TIMESTEPS, 0]))
x_val = np.array(x_val)
y_val = np.array(y_val)

for i in range(TRAIN_SIZE+VAL_SIZE, TRAIN_SIZE+VAL_SIZE+TEST_SIZE):
    x_test.append(np.array(data[i*SAMPLE_RATE:i*SAMPLE_RATE+TIMESTEPS]))
    y_test.append(np.array(data[i*SAMPLE_RATE+TIMESTEPS, 0]))
x_test = np.array(x_test)
y_test = np.array(y_test)
y_total = np.concatenate([y_train, y_val, y_test], axis=None)


'''
In this part network is implemented 
'''
# parameters
BATCH_SIZE = 10
NUM_EPOCHS = 20
HIDDEN_SIZE = 40

# keras modeling
model = Sequential()
model.add(LSTM(HIDDEN_SIZE,  input_shape=(TIMESTEPS, 8), return_sequences=True,activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.0, recurrent_dropout=0.0))
model.add(LSTM(HIDDEN_SIZE, return_sequences=False,activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.0, recurrent_dropout=0.0))
# model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')


print(model.summary())
history = model.fit(x_train, y_train, BATCH_SIZE=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=(x_val, y_val))

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


print('train loss history:', history.history["loss"])
print('test loss history:', history.history["val_loss"])
scores = model.evaluate(x_test, y_test, BATCH_SIZE=BATCH_SIZE, verbose=1)
print('\nTest result loss: %.3f' % (scores))

x_old =np.concatenate((x_train, x_val), axis=0)
y_pred = model.predict(x_test, BATCH_SIZE=BATCH_SIZE)
y_old = model.predict(x_old, BATCH_SIZE=BATCH_SIZE)

xc = range(NUM_EPOCHS)
plt.figure(1, figsize=(7, 5))
plt.subplot(211)
plt.plot(range(0, TRAIN_SIZE+VAL_SIZE), y_old, c='r', linewidth=3.0)
plt.plot(range(TRAIN_SIZE+VAL_SIZE, TRAIN_SIZE+VAL_SIZE+TEST_SIZE), y_pred, c='b', linewidth=3.0)
plt.plot(range(0, TRAIN_SIZE+VAL_SIZE+TEST_SIZE), y_total, 'c')
plt.xlabel('data')
plt.ylabel('value')
plt.title('LSTM ')
plt.grid(True)
plt.legend(['prediction for train', 'prediction for test',' exact'])
plt.style.use(['classic'])


plt.subplot(212)
plt.plot(range(TRAIN_SIZE+VAL_SIZE, TRAIN_SIZE+VAL_SIZE+TEST_SIZE), y_pred, 'b', linewidth=3.0)
plt.plot(range(TRAIN_SIZE+VAL_SIZE, TRAIN_SIZE+VAL_SIZE+TEST_SIZE), y_total[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE], 'c')
plt.xlabel('data')
plt.ylabel('value')
plt.title('value')
plt.grid(True)
plt.legend(['prediction for test',' exact'])
plt.style.use(['classic'])
plt.show()
