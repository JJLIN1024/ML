import numpy as np
import matplotlib.pyplot as plt


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1


# print(AND(1, 1))

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1


# print(NAND(1, 1))


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1


# print(OR(1, 1))


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


# print(XOR(1, 0))


def step_function(x):
    return np.array(x > 0, dtype=np.int)


# print(step_function(np.array([-1.0, 2.0])))
# --> [False  True] --> [0 1]
# print(step_function(np.array([1.0, 2.0])))
# --> [True  True] --> [1 1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()


def relu(x):
    return np.maximum(0, x)


# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-1.0, 5.0)
# plt.show()

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# loss function
# the smaller the better
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # one-hot
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # softmax --> 機率


def mean_square_error(y, t):
    return np.sum((y - t)**2) / 2


# print(mean_square_error(np.array(y), np.array(t)))


def cross_entropy_error(y, t):
    delta = 1e-7  # np.log(0) is -inf, so add delta to it to avoid this.
    return -np.sum(t * np.log(y + delta))


# print(cross_entropy_error(np.array(y), np.array(t)))


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y
