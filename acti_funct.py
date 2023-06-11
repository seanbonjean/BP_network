import numpy as np


def ReLU(input: float) -> float:
    """ReLU激活函数的实现"""
    if input < 0:
        return 0
    else:
        return input


def ReLU_deri(input: float) -> float:
    """ReLU激活函数的导数"""
    if input < 0:
        return 0
    else:
        return 1


def ReLU_all(input):
    """将输入Z矩阵的各元素依次通过ReLU"""
    output = np.empty_like(input)
    for i in range(input.size):
        output[i] = ReLU(input[i])
    return output


def sigmoid(input: float) -> float:
    """sigmoid激活函数的实现"""
    if input < 0:
        return 1 - 1 / (1 + np.exp(input))
    return 1 / (1 + np.exp(-input))


def sigmoid_deri(input: float) -> float:
    """sigmoid激活函数的导数"""
    sig = sigmoid(input)
    return sig * (1 - sig)


def sig_all(input):
    """将输入Z矩阵的各元素依次通过sigmoid"""
    output = np.empty_like(input)
    for i in range(input.size):
        output[i] = sigmoid(input[i])
    return output
