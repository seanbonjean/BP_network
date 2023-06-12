import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

from acti_funct import *


def printIMG(num: int) -> None:
    """显示图片"""
    img = np.reshape(vali_img[num, :], (28, 28))
    plt.matshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


def getAns(Y) -> int:
    """根据输出层结果判断该网络给出的答案"""
    ans = -1
    max = 0
    for i in range(10):
        if Ys[i] > max:
            max = Ys[i]
            ans = i
    return ans


# 读取数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

vali_img = mnist.validation.images  # 验证数据
vali_label = mnist.validation.labels  # 验证数据的标签

test_img = mnist.test.images  # 测试数据
test_label = mnist.test.labels  # 测试数据的标签

vali_num = mnist.validation.num_examples  # 验证数据的个数
test_num = mnist.test.num_examples  # 测试数据的个数

# 读取训练后的权值和偏置
weight1 = np.loadtxt('weights/trained/weight1.txt')
bias1 = np.loadtxt('weights/trained/bias1.txt')
weight2 = np.loadtxt('weights/trained/weight2.txt')
bias2 = np.loadtxt('weights/trained/bias2.txt')
weight3 = np.loadtxt('weights/trained/weight3.txt')
bias3 = np.loadtxt('weights/trained/bias3.txt')

# 测试
print()
print("Choose which mode u want to evaluate the weights: ")
print("1. validate random dataset and display corresponding raw-image")
print("2. test every dataset and get the accuracy")
print("Input ur choice: ", end='')
mode = int(input())

if mode == 1:
    vali_continue = True
    while vali_continue:
        which_vali = random.randint(0, vali_num-1)
        # 正向计算
        Xs = vali_img[which_vali, :]
        Z1s = np.add(np.dot(weight1, Xs), bias1)
        A1s = sig_all(Z1s)
        Z2s = np.add(np.dot(weight2, A1s), bias2)
        A2s = ReLU_all(Z2s)
        Z3s = np.add(np.dot(weight3, A2s), bias3)
        Ys = sig_all(Z3s)

        label = vali_label[which_vali]
        ans = getAns(Ys)

        print("The label is: " + str(label))
        print("The neural-network gives: " + str(ans))
        print("That is " + ("right" if label == ans else "wrong"))
        printIMG(which_vali)

        print("Continue?(Y/n): ", end='')
        if not input() == 'Y':
            vali_continue = False

elif mode == 2:
    right_count = 0
    for i in range(test_num):
        print(f"No.{i: >5}", end=' ')
        # 正向计算
        Xs = test_img[i, :]
        Z1s = np.add(np.dot(weight1, Xs), bias1)
        A1s = sig_all(Z1s)
        Z2s = np.add(np.dot(weight2, A1s), bias2)
        A2s = ReLU_all(Z2s)
        Z3s = np.add(np.dot(weight3, A2s), bias3)
        Ys = sig_all(Z3s)

        label = test_label[i]
        ans = getAns(Ys)

        if label == ans:
            right_count += 1
            print("RIGHT: label " + str(label) + ', ' + "ans " + str(ans))
        else:
            print("WRONG: label " + str(label) + ', ' + "ans " + str(ans))
    print()
    print("overall accuracy: " + str(right_count/test_num*100) + '%')
