import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

from acti_funct import *

ALPHA = 0.007  # 学习率alpha

# 读取数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

train_img = mnist.train.images  # 训练数据
train_label = mnist.train.labels  # 训练数据的标签

train_num = mnist.train.num_examples  # 训练数据的个数

# 读取初始权值和偏置
weight1 = np.loadtxt('weights/original/rand-weight1.txt')
bias1 = np.loadtxt('weights/original/rand-bias1.txt')
weight2 = np.loadtxt('weights/original/rand-weight2.txt')
bias2 = np.loadtxt('weights/original/rand-bias2.txt')
weight3 = np.loadtxt('weights/original/rand-weight3.txt')
bias3 = np.loadtxt('weights/original/rand-bias3.txt')

# 训练
for i in range(train_num):
    # 正向计算
    Xs = train_img[i, :]  # 输入层，为28x28像素图片展平后的784维向量（MNIST原始数据就是展平了的）
    Z1s = np.add(np.dot(weight1, Xs), bias1)  # 第一层的Z = WX+b
    A1s = sig_all(Z1s)  # 第一层的A = f(Z)，f为激活函数，此处为sigmoid
    Z2s = np.add(np.dot(weight2, A1s), bias2)  # 第二层的Z = WX+b
    A2s = ReLU_all(Z2s)  # 第二层的A = f(Z)，f为激活函数，此处为ReLU
    Z3s = np.add(np.dot(weight3, A2s), bias3)  # 第三层的Z = WX+b
    Ys = sig_all(Z3s)  # 第三层的A = f(Z)，f为激活函数，此处为sigmoid

    # 计算代价
    Cost = 0
    delta_Ys = np.empty_like(Ys)  # (yi - Yi)的向量，i=0~9
    for j in range(10):
        delta_Ys[j] = Ys[j] - (1 if j == train_label[i] else 0)  # 计算(yi - Yi)
        Cost += delta_Ys[j]**2  # 累加进代价E（也就是Cost）
    Cost /= 2  # E = (1/2)*(yi - Yi)^2
    print('Cost: '+str(Cost))  # 输出实时代价

    # 反向传播
    omiga3 = np.empty_like(Ys)  # omiga3 = (yi - Yi)*sigmoid'(Z3)
    for j in range(10):
        omiga3[j] = delta_Ys[j] * sigmoid_deri(Z3s[j])

    omiga2 = np.empty_like(A2s)  # omiga2 = sum(omiga3*W3)*ReLU'(Z2)
    for j in range(16):
        deri_EtoA = 0
        for k in range(10):
            deri_EtoA += omiga3[k] * weight3[k, j]
        omiga2[j] = deri_EtoA * ReLU_deri(Z2s[j])

    omiga1 = np.empty_like(A1s)  # omiga1 = sum(omiga2*W2)*sigmoid'(Z1)
    for j in range(16):
        deri_EtoA = 0
        for k in range(16):
            deri_EtoA += omiga2[k] * weight2[k, j]
        omiga1[j] = deri_EtoA * sigmoid_deri(Z1s[j])

    # 更新权重
    # dE/dw = omiga(m)*a(m-1)
    # dE/db = omiga(m)
    # w新 = w旧 - alpha*dE/dw
    # b新 = b旧 - alpha*dE/db
    for i in range(16):
        bias1[i] -= ALPHA * omiga1[i]
        for j in range(784):
            weight1[i, j] -= ALPHA * omiga1[i] * Xs[j]

    for i in range(16):
        bias2[i] -= ALPHA * omiga2[i]
        for j in range(16):
            weight2[i, j] -= ALPHA * omiga2[i] * A1s[j]

    for i in range(10):
        bias3[i] -= ALPHA * omiga3[i]
        for j in range(16):
            weight3[i, j] -= ALPHA * omiga3[i] * A2s[j]

# 保存训练结果
np.savetxt('weights/trained/weight1.txt', weight1)
np.savetxt('weights/trained/bias1.txt', bias1)
np.savetxt('weights/trained/weight2.txt', weight2)
np.savetxt('weights/trained/bias2.txt', bias2)
np.savetxt('weights/trained/weight3.txt', weight3)
np.savetxt('weights/trained/bias3.txt', bias3)
