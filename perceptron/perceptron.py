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

    def train(self, X, Y, epochs=100, batch_size=256, learning_rate=0.001):
        # 获取特征矩阵的行列数，行数m表示特征向量的长度，列数n表示样本的个数
        m, n = X.shape
        # 随机初始化权重向量
        self.w = np.random.rand(1, m)
        # 初始化偏置为0
        self.b = 0

        # 进行max_iter次迭代,其实这里应该是 while True，但是有可能一直不能找到完全分离的超平面导致无法训练完成
        for epoch in range(epochs):
            print("Iter: ", epoch + 1)
            # 对于每一个样本都进行梯度下降
            for i in range(n):
                # 获取第i个样本
                x_i = X[:, i:i+1]
                y_i = Y[i]
                # 判断是否为误分类样本, 公式为：-y_i(w*x_i+b) >= 0
                if -y_i * (np.dot(self.w, x_i) + self.b) >= 0:
                    # 对于误分类点进行梯度下降，更新参数
                    self.w += np.transpose(learning_rate * y_i * x_i)
                    self.b += (learning_rate * y_i)

    def test(self, test_features, test_labels):
        predict_labels = np.dot(self.w, test_features) + self.b
        predict_labels[predict_labels > 0] = 1
        predict_labels[predict_labels < 0] = -1
        result = np.equal(predict_labels, test_labels)
        correct_count = np.sum(result)
        print("Accuracy: ", correct_count / result.shape[1])


class DualPerceptron(object):
    """感知机对偶形式"""
    pass


def load_data():
    """加载数据集"""
    train_features, train_labels, test_features, test_labels = load_minist()

    # 由于感知机是二分类任务，故将标签中0~4设置为负类-1，将标签5～9设置为正类+1
    train_labels[train_labels <= 4] = -1
    train_labels[train_labels >= 5] = 1
    test_labels[test_labels <= 4] = -1
    test_labels[test_labels >= 5] = 1

    # 对特征数据做一个简单的归一化
    train_features = np.true_divide(train_features, 255)
    test_features = np.true_divide(test_features, 255)

    # 对特征数据做转置，即一列代表一个样本
    train_features = np.transpose(train_features)
    test_features = np.transpose(test_features)

    return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = load_data()
    perceptron = Perceptron()
    perceptron.train(train_features, train_labels)
    perceptron.test(test_features, test_labels)