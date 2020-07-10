import time

import numpy as np
from datasets.minist.loader import load_minist


class NavieBayes(object):
    def __init__(self):
        # 每个类别的概率(先验概率)，key是类别，value是其概率
        self.class_probability = {}
        # 在特定类别前提下的各个特征值的概率(条件概率)
        # key是一个三元组 (feature_index, feature_value, class_label)，value是其概率
        # 其中，feature_index表示是第几个特征，feature_value是特征值，class_label是类别前提
        self.feature_class_probability = {}
        self.log_feature_class_probability = {}
        # 特征维度
        self.feature_num = 0
        # 贝叶斯估计的lambda常数
        self.l = 1
        # 在训练集中每一个维度下出现的特征可能的取值
        # key是一个二元组 (class_label, feature_index)，value是特征可能的取值的数组
        self.feature_index_values = {}

    def fit(self, x, y, l=1):
        """
        训练朴素贝叶斯模型
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :param y: 标签数组
        :param l: 贝叶斯估计的大于等于零的lambda常数，常取1
        :return:
        """
        # 记录开始训练的时间
        print("Starting train NavieBayes model...")
        start = time.process_time()

        # 保存训练时使用的贝叶斯估计的lambda常数，在预测时还要用到
        self.l = l

        # 统计出各个类别标签的出现的次数，unique是类别标签数组，count是对应标签出现的次数数组
        class_labels, class_counts = np.unique(y, return_counts=True)
        # 统计类别标签的出现次数的总数
        class_total = np.sum(class_counts)
        # 计算每个类别出现的概率(式4.11)
        for class_label, class_count in zip(class_labels, class_counts):
            # 这里做一个处理：为了防止某个标签没有对应的样本，在分子上加上l，在分母上加上l*k
            # 其中l是大于等于零的lambda常数，常取1；k是类别标签的数量
            # 注意：这里可以对概率使用log函数进行处理，防止在预测时各个概率连乘之后的最终概率值过小导致下溢
            self.class_probability[class_label] = np.log((class_count + l) / (class_total + l * len(class_labels)))

        # 获取特征维度
        self.feature_num = x.shape[0]

        # 根据公式（式4.10）计算条件概率 P(X=x|Y=y)
        # 遍历所有的类别标签
        for class_label, class_count in zip(class_labels, class_counts):
            # 获取该类别标签下的所有的样本
            samples = x[:, np.where(y == class_label)[0]]
            # 对特征的每一个维度进行遍历
            for feature_index in range(self.feature_num):
                # 获取该类别标签下所有样本的第feature_index维的特征值
                features = samples[feature_index]
                # 统计该维度下特征的可能取值和其出现次数
                feature_values, feature_value_counts = np.unique(features, return_counts=True)
                # 记录下在该类别特征和该维度下特征可能的取值集合
                self.feature_index_values[f"{class_label}, {feature_index}"] = feature_values
                # 计算在该类别和维度下的各个特征值的概率
                for feature_value, feature_value_count in zip(feature_values, feature_value_counts):
                    # key是一个三元组(feature_index, feature_value, class_label)
                    key = f"{feature_index}, {feature_value}, {class_label}"
                    # 计算其概率值，分子加上常数l，分母加上S_j*l
                    # 其中，S_j表示特征可能取值的个数
                    # 原公式中是没有最后的+1的，这里是考虑到如果某一个维度的都是同一个值，但是在测试集中出现了其他值的情况
                    # 即认为一个特征值对应的类别不能是百分之百的
                    value = (feature_value_count + l) / (class_count + l * (len(feature_values) + 1))
                    self.feature_class_probability[key] = value
                    self.log_feature_class_probability[key] = np.log(value)

        # 记录训练结束的时间
        end = time.process_time()
        print(f"Training NavieBayes model complete, cost time {end - start}s")

    def predict(self, x):
        """
        预测输入的实例的类别
        :param x: 实例特征
        :return: 预测的标签和每一个类别的概率数组
        """
        # 所有类别的预测概率数组
        predicted_probability = np.zeros(len(self.class_probability.keys()))
        # 预测的标签
        predicted_label = None

        # 预测的标签的概率, 初始化为无穷小
        predicted_label_probability = float('-inf')
        # 遍历每一个类别
        for index, (class_label, class_probability) in enumerate(self.class_probability.items()):
            # 该类别下的预测概率
            temp_probability = class_probability
            # 遍历每一个特征
            for feature_index in range(self.feature_num):
                # 获取第feature_index个特征的值
                feature_value = x[feature_index]
                # 从保存的字典中获取对应的在该类别前提下的该特征值的概率
                key = f"{feature_index}, {feature_value}, {class_label}"
                value = self.log_feature_class_probability.get(key, 0)
                # 如果概率为0，表示这个特征值没有在训练集中出现,需要用原始的概率去计算，这里也可以使用一个极小的数来代替
                if value == 0:
                    # # 通过原始概率求得这个未出现过的特征值的估计概率
                    # temp_value = 1
                    # for other_feature_value in self.feature_index_values[f"{class_label}, {feature_index}"]:
                    #     temp_key = f"{feature_index}, {other_feature_value}, {class_label}"
                    #     temp_value -= self.feature_class_probability[temp_key]
                    # # 对概率使用log函数进行处理
                    # value = np.log(temp_value)

                    value = np.log(0.000001)
                # 如果概率使用了log函数进行处理，这里进行累加即可
                temp_probability += value
            # 如果当前预测的概率大于预测标签的概率，更新预测的标签
            if temp_probability > predicted_label_probability:
                predicted_label = class_label
                predicted_label_probability = temp_probability
            # 记录预测结果
            predicted_probability[index] = temp_probability

        return predicted_label, predicted_probability

    def test(self, x, y):
        """
        测试朴素贝叶斯模型
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :param y: 标签数组
        :return: 准确率accuracy
        """
        # 记录测试用时
        print(f"Starting test, the size of test set is {x.shape}...")
        start = time.process_time()

        err = 0
        for i in range(x.shape[1]):
            predicted_label, predicted_probability = self.predict(x[:, i])
            if predicted_label != y[i]:
                err += 1
        end = time.process_time()
        print(f"Testing complety, cost time {end-start}s.")

        return 1 - (err / x.shape[1])


def load_data():
    """加载minist数据集"""
    train_images, train_labels, test_images, test_labels = load_minist()

    # 对特征数据做转置，即一列代表一个样本
    train_images = np.transpose(train_images)
    test_images = np.transpose(test_images)

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    data = load_data()
    clf = NavieBayes()
    clf.fit(data[0], data[1])
    print("Accuracy on test set: ", clf.test(data[2], data[3]))
