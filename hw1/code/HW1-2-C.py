"""
HW1 Part 2-C: Distinguish between fruits
Learning method adopted is based on "Perceptron Network"
Meysam Parvizi
"""

import numpy as np
import matplotlib.pyplot as plt

# -1 = soft,   1 = hard
# -1 = circle, 1 = non-circle

# [
#   [hardness],
#   [shape]
# ]
training_input = np.array([
    [
        [-1],
        [-1]
    ],
    [
        [1],
        [-1]
    ],
    [
        [-1],
        [1]
    ]
])

# apple, orange, pear (1-hot output)
training_output = np.array([
    [
        [1],
        [-1],
        [-1]
    ],
    [
        [-1],
        [1],
        [-1]
    ],
    [
        [-1],
        [-1],
        [1]
    ]
])

input_size = 2
output_size = 3


def neural_network(w, x, b, theta):
    # multiply input vector x by weights matrix w and add bias vector b to it
    net = np.dot(w, x) + b
    return activation_function(net, theta)


def activation_function(x, theta):
    # convert theta to a replicated array with the same shape of x
    theta = np.reshape(np.repeat(theta, np.prod(np.shape(x))), np.shape(x))
    # output sign of x - theta
    return np.sign(x - theta)


def train():

    # Step 0
    w = np.zeros((output_size, input_size))
    b = np.zeros((output_size, 1))
    theta = 0
    learning_rate = 1

    epochs = 0
    costs = []

    is_total_error_nonzero = np.ones((output_size, 1), dtype=int)

    while True:
        epochs += 1
        # Step 1
        total_error = np.zeros((output_size, 1))
        for i in range(len(training_input)):
            # Step 2
            # set activations of input units
            x = training_input[i]
            t = training_output[i]
            # Step 3
            # compute response of output units
            h = neural_network(w, x, b, theta)
            # Step 4
            error = h - t
            # update a binary array indicating which outputs have non-zero error in current epoch
            is_error_nonzero = 1 * (error != 0)
            # update weights and bias if an error occurred for this pattern
            # THIS IS DONE ACCORDING TO THE VALUES OF ELEMENTS OF is_error_nonzero AND is_total_error_nonzero
            x_t = np.transpose(training_input[i])
            w = w + learning_rate * \
                np.dot(np.multiply(np.multiply(
                    is_error_nonzero, is_total_error_nonzero), t), x_t)
            b = b + learning_rate * \
                np.multiply(np.multiply(
                    is_error_nonzero, is_total_error_nonzero), t)
            # calculate sum of errors absolute values to check whether all errors are zero or not
            total_error += np.abs(error)

        costs.append(np.sum(total_error))
        # update a binary array indicating which outputs have non-zero error in all epochs
        is_total_error_nonzero *= (1 * (total_error != 0))
        # Step 5
        if np.all(is_total_error_nonzero == 0):
            break
    return epochs, costs, w, b


epochs, costs, w, b = train()

# change training_input[] index to indicate the neural network input
index = np.random.randint(len(training_input))
x = training_input[index]
result = neural_network(w, x, b, 0)

print('Training finished after ' + str(epochs) + ' epochs')
print('Output is \r\n' + str(result) + '\r\n for the input \r\n' + str(x))
ep = np.arange(1, epochs+1, 1, dtype=int)
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
