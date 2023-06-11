import random
import numpy as np


def rand_gen() -> int:
    # 生成一个-1~1范围内的浮点数
    return random.uniform(-1, 1)


print("WARNING: This action will overwrite the old version of randomly generated weights. Continue?(Y/n)", end='')
if input() == 'Y':
    # 创建空矩阵存放权值和偏置
    weight1 = np.empty([16, 784], dtype=float)
    weight2 = np.empty([16, 16], dtype=float)
    weight3 = np.empty([10, 16], dtype=float)
    bias1 = np.empty([16], dtype=float)
    bias2 = np.empty([16], dtype=float)
    bias3 = np.empty([10], dtype=float)

    # 随机生成权值和偏置
    for i in range(16):
        bias1[i] = rand_gen()
        for j in range(784):
            weight1[i, j] = rand_gen()

    for i in range(16):
        bias2[i] = rand_gen()
        for j in range(16):
            weight2[i, j] = rand_gen()

    for i in range(10):
        bias3[i] = rand_gen()
        for j in range(16):
            weight3[i, j] = rand_gen()

    # 保存随机生成的初始权值和偏置
    np.savetxt('weights/original/rand-weight1.txt', weight1)
    np.savetxt('weights/original/rand-bias1.txt', bias1)
    np.savetxt('weights/original/rand-weight2.txt', weight2)
    np.savetxt('weights/original/rand-bias2.txt', bias2)
    np.savetxt('weights/original/rand-weight3.txt', weight3)
    np.savetxt('weights/original/rand-bias3.txt', bias3)
