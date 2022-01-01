import numpy as np
import matplotlib.pyplot as plt

data = np.load('Alphabets.npy')
labels = np.load('Alphabet_labels.npy')
alphabets =  np.array(['C','I','O','P','S','U','X','Z'])
for i in range(0,500):
    plt.imshow(data[i].reshape(28,28))
    print(alphabets[labels[i]])
    plt.show()
print(data.shape,labels.shape)