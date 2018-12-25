"""
HW1 Part 1: Distinguish character "U" from others
Learning method adopted is based on "Perceptron Network"
Meysam Parvizi
"""

import matplotlib.pyplot as plt
import numpy as np

training_input = np.array([
    # U
    [
        1, 0, 1,
        1, 0, 1,
        1, 1, 1
    ],

    # I
    [
        0, 1, 0,
        0, 1, 0,
        0, 1, 0
    ],

    # O
    [
        1, 1, 1,
        1, 0, 1,
        1, 1, 1
    ],

    # L
    [
        0, 1, 0,
        0, 1, 0,
        0, 1, 1
    ]
])

training_output = np.array([1, -1, -1, -1])

input_size = 9
output_size = 1


def neural_network(w, x, b, theta):
    # multiply input vector x by weights matrix w and add bias vector b to it
    net = np.dot(w, x) + b
    return activation_function(net, theta)


def activation_function(x, theta):
    if x > theta:
        return 1
    elif x < -theta:
        return -1
    else:
        return 0


def train():

    # Step 0
    w = np.zeros((output_size, input_size))
    b = np.zeros((output_size, 1))
    theta = 0
    learning_rate = 1

    epochs = 0
    costs = []
    while True:
        epochs += 1
        # Step 1
        total_error = 0
        for i in range(len(training_input)):
            # Step 2
            x = training_input[i]
            t = training_output[i]
            # Step 3
            h = neural_network(w, x, b, theta)
            # Step 4
            error = h - t
            if error != 0:
                w = w + learning_rate * x * t
                b = b + learning_rate * t

            # calculate sum of errors absolute values to check whether all errors are zero or not
            total_error += np.abs(error)

        costs.append(total_error)
        # Step 5
        if total_error == 0:
            break
    return epochs, costs, w, b


epochs, costs, w, b = train()

# change training_input[] index to indicate the neural network input character
index = np.random.randint(len(training_input))
x = training_input[index]
result = neural_network(w, x, b, 0)

print('Training finished after ' + str(epochs) + ' epochs')
print('Output is [' + str(result) + '] for the input \r\n' + str(np.reshape(x,(3,3))))
if result == 1:
    print("It's U")
else:
    print("It's NOT U")

ep = np.arange(1, epochs+1, 1)
plt.figure(1, figsize=(10, 6))
plt.plot(ep, costs, color='c', linewidth=4)
plt.xlim(1, epochs)
plt.xticks(ep)
plt.ylim(0, np.max(costs))
plt.title('Loss Function', fontsize=20)
plt.xlabel(r'$Epochs$', fontsize=15)
plt.ylabel(r'$Cost$', fontsize=15)
plt.grid(True)
plt.show()
