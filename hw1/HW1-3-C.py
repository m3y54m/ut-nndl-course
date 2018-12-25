"""
HW1 Part 3-C: Classification of points using Bi-polar Sigmoid Function
Meysam Parvizi
"""

import matplotlib.pyplot as plt
import numpy as np

# Q1 is number of points in class 1
# Q2 is number of points in class 2
#Q1, Q2 = 100, 100
Q1, Q2 = 1000, 10
# total number of training points
Q = Q1 + Q2

input_size = 2
output_size = 1

gamma = 10
learning_rate = 0.01
error_threshold = 0.5

q1_inputs = 1 + 0.5 * np.random.randn(Q1, input_size, 1)
q1_outputs = np.ones((Q1, output_size, 1))

q2_inputs = -1 + 0.5 * np.random.randn(Q2, input_size, 1)
q2_outputs = -1 * np.ones((Q2, output_size, 1))

training_input = np.concatenate((q1_inputs, q2_inputs), axis=0)
training_output = np.concatenate((q1_outputs, q2_outputs), axis=0)

x1_q1 = np.squeeze(q1_inputs, axis=2)[:, 0]
x2_q1 = np.squeeze(q1_inputs, axis=2)[:, 1]

x1_q2 = np.squeeze(q2_inputs, axis=2)[:, 0]
x2_q2 = np.squeeze(q2_inputs, axis=2)[:, 1]


def neural_network(w, x, b, gamma):
    # multiply input vector x by weights matrix w and add bias vector b to it
    net = np.dot(w, x) + b
    h = activation_function(net, gamma)
    h_prime = activation_function_derivative(net, gamma)
    return h, h_prime


def activation_function(x, gamma):
    # output tanh of gamma * x
    return np.tanh(gamma * x)


def activation_function_derivative(x, gamma):
    return (1 - np.tanh(gamma * x)**2)


# train using SIGMOID
def train():

    # Step 0
    w = np.zeros((output_size, input_size))
    b = np.zeros((output_size, 1))

    epochs = 0
    costs = []

    is_total_error_gt_thresh = np.ones((output_size, 1), dtype=int)

    timeout = False

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
            # compute response of output units and their derivatives
            h, h_prime = neural_network(w, x, b, gamma)
            # Step 4
            error = t - h
            # update weights and bias if an error occurred for this pattern
            # THIS IS DONE ACCORDING TO THE VALUES OF ELEMENTS OF is_total_error_gt_thresh
            x_t = np.transpose(training_input[i])
            w = w + (learning_rate * gamma) * \
                np.dot((is_total_error_gt_thresh *
                        np.multiply(error, h_prime)), x_t)
            b = b + (learning_rate * gamma) * \
                (is_total_error_gt_thresh * np.multiply(error, h_prime))
            # calculate sum of errors absolute values to check whether all errors are less than threshold or not
            total_error += np.abs(error)

        costs.append(np.sum(total_error))
        # update a binary array indicating which outputs have error greater than threshold in all epochs
        is_total_error_gt_thresh *= (1 * (total_error >= error_threshold))
        # Step 5
        if np.all(is_total_error_gt_thresh == 0):
            break

        if epochs >= 1000:
            timeout = True
            break

    return timeout, epochs, costs, w, b


timeout, epochs, costs, w, b = train()

if timeout:
    print('Timeout error')
else:
    # change training_input[] index to indicate the neural network input character
    index = np.random.randint(Q)
    x = training_input[index]
    result, _ = neural_network(w, x, b, gamma)
    result = np.sign(result)
    print('Training finished after ' + str(epochs) + ' epochs')
    print('Output is [' + str(np.squeeze(result)) +
          '] for the input ' + str(np.squeeze(x)))

    x1_line = np.linspace(-3, 3, 10)

    if (w[0, 1] != 0):
        x2_line = (-w[0, 0] * x1_line - b[0, 0]) / w[0, 1]
    else:
        x1_line = np.repeat(-b[0, 0] / w[0, 0], 10)
        x2_line = np.linspace(-3, 3, 10)

    plt.figure(1, figsize=(10, 10))
    plt.scatter(x1_q1, x2_q1, marker='s', c='none',
                edgecolors='b', label="Class 1")
    plt.scatter(x1_q2, x2_q2, marker='o', c='none',
                edgecolors='r', label="Class 2")
    plt.plot(x1_line, x2_line, color='c',
             linewidth=4, label='Separating Line')
    plt.plot(x[0, 0], x[1, 0], color='none',
             marker='x', markersize=10, markeredgewidth=4, markeredgecolor='c', label='Test Point')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title('Classification Using Bipolar Sigmoid Function', fontsize=20)
    plt.xlabel(r'$X_1$', fontsize=15)
    plt.ylabel(r'$X_2$', fontsize=15)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    ep = np.arange(1, epochs+1, 1, dtype=int)
    plt.figure(2, figsize=(10, 6))
    plt.plot(ep, costs, color='c', linewidth=4)
    plt.xlim(1, epochs)
    plt.ylim(0, np.max(costs))
    plt.title('Loss Function', fontsize=20)
    plt.xlabel(r'$Epochs$', fontsize=15)
    plt.ylabel(r'$Cost$', fontsize=15)
    plt.grid(True)
    plt.show()
