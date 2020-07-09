import time

import numpy as np
from datasets.minist.loader import load_minist


class NavieBayes(object):
    def __init__(self):
        # 每个类别的概率，key是类别，value是其概率
        self.class_probability = {}
        # 在特定类别前提下的各个特征值的概率
        # key是一个三元组 (feature_index, feature_value, class_label)，value是其概率
        # 其中，feature_index表示是第几个特征，feature_value是特征值，class_label是类别前提
        self.feature_class_probability = {}
        # 特征数量
        self.feature_num = 0

    def fit(self, x, y):
        """
        训练朴素贝叶斯模型
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :param y: 标签数组
        :return:
        """
        # 记录开始训练的时间
        print("Starting train NavieBayes model...")
        start = time.process_time()

        # 统计出各个类别标签的出现的次数，unique是类别标签数组，count是对应标签出现的次数数组
        class_labels, class_counts = np.unique(y, return_counts=True)
        # 统计类别标签的出现次数的总数
        class_total = np.sum(class_counts)
        # 计算每个类别出现的概率
        for class_label, class_count in zip(class_labels, class_counts):
            self.class_probability[class_label] = class_count / class_total

        # 得到特征数量feature_num
        self.feature_num = x.shape[0]

        # 计算在每个类别下各个特征值的概率
        for i in range(len(class_labels)):
            # 第i个类别标签
            class_label = class_labels[i]
            # 第i个类别标签的出现次数
            class_count = class_counts[i]

            # 获取第i个类别标签的下标
            class_index = np.where(y == class_label)
            # 筛选出第i个类别标签对应的所有实例样本
            samples = x[:, class_index[0]]

            # 遍历每一个特征,计算出在特定类别前提下的各个特征值的概率
            for feature_index in range(self.feature_num):
                # 获取第feature_index个特征
                feature = samples[feature_index:feature_index+1, :]
                # 统计这个特征的所有可能值和其对应出现的次数
                feature_values, feature_counts = np.unique(feature, return_counts=True)
                # 计算出在特定类别前提下的各个特征值的概率
                for feature_value, feature_count in zip(feature_values, feature_counts):
                    # key是一个三元组(feature_index, feature_value, class_label)
                    key = f"{feature_index}, {feature_value}, {class_label}"
                    value = feature_count / class_count
                    self.feature_class_probability[key] = value

        # 记录训练结束的时间
        end = time.process_time()
        print(f"Training NavieBayes model complete, cost time {end-start}s")

    def predict(self, x):
        """
        预测输入的实例的类别
        :param x: 实例特征
        :return: 每一个类别的概率数组
        """
        # 预测结果,key是类别，value是预测的概率
        predicted = {}
        # 预测的标签
        predicted_label = None

        # 预测的标签的概率, 初始化为无穷小
        predicted_label_probability = float('-inf')
        # 遍历每一个类别
        for class_label, class_probability in self.class_probability.items():
            # 该类别下的预测概率
            predicted_probability = class_probability
            # 遍历每一个特征
            for feature_index in range(self.feature_num):
                # 获取第feature_index个特征的值
                feature_value = x[feature_index]
                # 从保存的字典中获取对应的在该类别前提下的该特征值的概率，并相乘
                key = f"{feature_index}, {feature_value}, {class_label}"
                predicted_probability *= self.feature_class_probability.get(key, 0)
            # 如果当前预测的概率大于预测标签的概率，更新预测的标签
            if predicted_probability > predicted_label_probability:
                predicted_label = class_label
                predicted_label_probability = predicted_probability
            # 记录预测结果
            predicted[class_label] = predicted_probability

        return predicted_label, predicted

    def test(self, x, y):
        """
        测试朴素贝叶斯模型
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :param y: 标签数组
        :return: 准确率accuracy
        """
        err = 0
        for i in range(x.shape[1]):
            predicted_label, predicted = self.predict(x[:, i])
            if predicted_label != y[i]:
                err += 1
        return 1 - (err / x.shape[1])


def load_data(normalize=True):
    """加载minist数据集"""
    train_images, train_labels, test_images, test_labels = load_minist()

    # 对特征数据做转置，即一列代表一个样本
    train_images = np.transpose(train_images)
    test_images = np.transpose(test_images)

    if normalize:
        # 对特征数据做一个简单的归一化,使其落入(0,1)的区间内，因为像素的的范围是0～255，故除以255即可
        train_images = np.true_divide(train_images, 255)
        test_images = np.true_divide(test_images, 255)

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    data = load_data(normalize=False)
    clf = NavieBayes()
    clf.fit(data[0], data[1])
    print("Accuracy on test set: ", clf.test(data[2], data[3]))
