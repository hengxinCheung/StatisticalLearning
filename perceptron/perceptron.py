# coding=utf-8
# author: hengxincheung
# date: 2020-07-06
import numpy as np
from datasets.minist.loader import load_minist


class Perceptron(object):
    """感知机原始形式"""
    def __init__(self):
        # 权重向量
        self.w = None
        # 偏置
        self.b = None

    def train(self, x, y, epochs=1, learning_rate=0.001):
        # 获取特征矩阵的行列数，行数m表示特征向量的长度，列数n表示样本的个数
        m, n = x.shape
        # 随机初始化权重向量
        self.w = np.random.rand(1, m)
        # 初始化偏置为0
        self.b = 0

        # 进行epochs次迭代,其实这里应该是 while True，但是有可能因为数据集不是线性可分的导致死循环
        for epoch in range(epochs):
            print("Epoch: ", epoch + 1)
            # 对于每一个样本进行梯度下降
            for i in range(n):
                # 获取第i个样本的数据
                x_i = x[:, i:i+1]
                y_i = y[i]
                # 根据公式 $-y_i(w \cdot x_i + b) > 0$ 为误分类点
                if -y_i*(np.dot(self.w, x_i) + self.b) >= 0:
                    # 更新权值和偏置
                    self.w += np.transpose(learning_rate * y_i * x_i)
                    self.b += learning_rate * y_i
            self.test(x, y)
        print("Training complete!")

    def test(self, x, y):
        predicted = -y * (np.dot(self.w, x) + self.b)
        err = predicted[predicted >= 0]
        print("Accuracy: ", 1 - (len(err) / predicted.shape[1]))


class DualPerceptron(object):
    """感知机对偶形式"""
    pass


def load_data():
    """加载数据集"""
    train_images, train_labels, test_images, test_labels = load_minist()

    # 由于感知机是二分类任务，故将标签中0~4设置为负类-1，将标签5～9设置为正类+1
    train_labels[train_labels <= 4] = -1
    train_labels[train_labels >= 5] = 1
    test_labels[test_labels <= 4] = -1
    test_labels[test_labels >= 5] = 1

    # 对特征数据做一个简单的归一化,使其落入(0,1)的区间内，因为像素的的范围是0～255，故除以255即可
    train_images = np.true_divide(train_images, 255)
    test_images = np.true_divide(test_images, 255)

    # 对特征数据做转置，即一列代表一个样本
    train_images = np.transpose(train_images)
    test_images = np.transpose(test_images)

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    data = load_data()
    clf = Perceptron()
    clf.train(data[0], data[1], epochs=10)
    clf.test(data[2], data[3])
