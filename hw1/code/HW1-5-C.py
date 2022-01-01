"""
HW1 Part 5-C: Classification of points using Perceptron and input transformation
Meysam Parvizi
"""

import matplotlib.pyplot as plt
import numpy as np

theta = 0
learning_rate = 1

training_input = np.array([
    [
        [1],
        [1]
    ],
    [
        [-1],
        [1]
    ],
    [
        [1],
        [-1]
    ],
    [
        [-1],
        [-1]
    ],
    [
        [0],
        [0]
    ],
    [
        [1],
        [0]
    ]
])

training_output = np.array([
    [
        [-1],
    ],
    [
        [-1],
    ],
    [
        [-1],
    ],
    [
        [-1],
    ],
    [
        [1],
    ],
    [
        [1],
    ]
])


x1_c1 = np.squeeze(training_input, axis=2)[0:4, 0]
x2_c1 = np.squeeze(training_input, axis=2)[0:4, 1]

x1_c2 = np.squeeze(training_input, axis=2)[4:6, 0]
x2_c2 = np.squeeze(training_input, axis=2)[4:6, 1]

# map training input points to another space
input_size = 1
output_size = 1

# transformation
z1 = training_input[:,1,:]**2

z_training_input = (z1)[:, :, np.newaxis]


def neural_network(w, x, b, theta):
    # multiply input vector x by weights matrix w and add bias vector b to it
    net = np.dot(w, x) + b
    return activation_function(net, theta)


def activation_function(x, theta):
    # convert theta to a replicated array with the same shape of x
    theta = np.reshape(np.repeat(theta, np.prod(np.shape(x))), np.shape(x))
    # output sign of x - theta
    return np.sign(x - theta)


# train using PERCEPTRON
def train():

    # Step 0
    w = np.zeros((output_size, input_size))
    b = np.zeros((output_size, 1))

    epochs = 0
    costs = []

    is_total_error_nonzero = np.ones((output_size, 1), dtype=int)

    timeout = False

    while True:
        epochs += 1
        # Step 1
        total_error = np.zeros((output_size, 1))
        for i in range(len(training_input)):
            # Step 2
            # set activations of input units
            z = z_training_input[i]
            t = training_output[i]
            # Step 3
            # compute response of output units
            h = neural_network(w, z, b, theta)
            # Step 4
            error = h - t
            # update a binary array indicating which outputs have non-zero error in current epoch
            is_error_nonzero = 1 * (error != 0)
            # update weights and bias if an error occurred for this pattern
            # THIS IS DONE ACCORDING TO THE VALUES OF ELEMENTS OF is_error_nonzero AND is_total_error_nonzero
            z_t = np.transpose(z_training_input[i])
            w = w + learning_rate * \
                np.dot(((is_error_nonzero * is_total_error_nonzero) * t), z_t)
            b = b + learning_rate * \
                ((is_error_nonzero * is_total_error_nonzero) * t)
            # calculate sum of errors absolute values to check whether all errors are zero or not
            total_error += np.abs(error)

        costs.append(np.sum(total_error))
        # update a binary array indicating which outputs have non-zero error in all epochs
        is_total_error_nonzero *= (1 * (total_error != 0))
        # Step 5
        if np.all(is_total_error_nonzero == 0):
            break

        if epochs >= 1000:
            timeout = True
            break

    return timeout, epochs, costs, w, b


timeout, epochs, costs, w, b = train()

if timeout:
    plt.close(1)
    print('Timeout error')
else:
    # change training_input[] index to indicate the neural network input character
    index = np.random.randint(len(training_input))
    x = training_input[index]
    z = z_training_input[index]
    result = neural_network(w, z, b, theta)
    print('Training finished after ' + str(epochs) + ' epochs')
    print('Output is [' + str(np.squeeze(result)) +
          '] for the input ' + str(np.squeeze(x)))

    x1, x2 = np.meshgrid(np.arange(-2, 2.5, 0.025), np.arange(-2, 2.5, 0.025))
    w1, w2 = w[0, 0], b[0,0]

    plt.figure(1, figsize=(10, 10))
    plt.scatter(x1_c1, x2_c1, marker='s', c='none',
                edgecolors='b', label="Class 1")
    plt.scatter(x1_c2, x2_c2, marker='o', c='none',
                edgecolors='r', label="Class 2")
    plt.contour(x1, x2, w1*(x2**2) + w2, [0], colors='c',
             linewidths=4)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Classification Using Perceptron Method', fontsize=20)
    plt.xlabel(r'$X_1$', fontsize=15)
    plt.ylabel(r'$X_2$', fontsize=15)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    ep = np.arange(1, epochs+1, 1, dtype=int)
    plt.figure(2, figsize=(10, 6))
    plt.plot(ep, costs, color='c', linewidth=4)
    plt.xlim(1, epochs)
    plt.xticks(ep)
    plt.ylim(0, np.max(costs))
    plt.title('Loss Function', fontsize=20)
    plt.xlabel(r'$Epochs$', fontsize=15)
    plt.ylabel(r'$Cost$', fontsize=15)
    plt.grid(True)
    plt.show()
