import time

import numpy as np
from datasets.minist.loader import load_minist


class DecisionTreeNode(object):
    """决策树节点"""
    pass


class BaseDecisionTree(object):
    """决策树基类，不直接使用"""

    # 支持使用的评判准则，key是评判准则的字符串名，value是允许调用的方法对象
    SUPPORTED_CRITERION = {}
    # 支持使用的调整拆分的方法，key是调整拆分的方法名，value是允许调用的方法对象
    SUPPORTED_SPLITTER = {}

    def __init__(self,
                 criterion,
                 splitter,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=65535):
        """
        决策树基类初始化方法
        :param criterion: 使用的评判准则，如信息增益、信息增益比、基尼指数等
        :param splitter: 调整拆分的方法
        :param min_samples_split: 调整拆分的最小的样本数
        :param min_impurity_split: 调整拆分的最小不纯度
        :param max_depth: 树生成的最大深度
        """
        self.criterion = self.SUPPORTED_CRITERION[criterion]
        self.splitter = self.SUPPORTED_SPLITTER[splitter]
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth

        # 决策树的根节点
        self.root = None

    def fit(self, x, y):
        """
        训练决策树
        :param x:
        :param y:
        :return:
        """
        pass

    def _build_tree(self, x, y, current_depth=0):
        """
        递归地构建决策树
        :param x:
        :param y:
        :param current_depth:
        :return:
        """

        # 获取特征个数和样本个数
        n_features, n_samples = np.shape(x)

        # 切分特征下标
        cut_feature_idx = None
        # 切分点
        cut_point = None
        # 当前最大的不纯度
        current_impurity = 0
        # 在所有可能的特征中遍历寻找切分特征和切分点
        for feature_idx in range(n_features):
            # 得到第 feature_idx 个特征
            feature = x[feature_idx, :]
            # 获取特征可能的值和其对应出现的次数
            feature_values, feature_value_counts = np.unique(feature, return_counts=True)
            # 在特征的所有可能的切分点（特征值）中遍历
            for feature_value, feature_value_count in zip(feature_values, feature_value_counts):
                # 根据特征值分割标签集
                y1 = y[np.where(feature == feature_value)]
                y2 = y[np.where(feature != feature_value)]
                # 根据评判准则计算不纯度
                impurity = self.criterion(y, y1, y2)
                # 如果不纯度大于当前的不纯度，记录当前的特征下标和特征值作为切分特征和切分点
                if impurity > current_impurity:
                    cut_feature_idx = feature_idx
                    cut_point = feature_value
                    current_impurity = impurity

        # 如果当前最大的不纯度小于调整拆分的最小不纯度，返回树节点
        if current_impurity < self.min_impurity_split:
            return DecisionTreeNode()
        # 否则，需要继续递归构造树节点
        pass

    def predict(self, x):
        """
        使用决策树进行单个样本的预测
        :param x:
        :return:
        """
        pass


class ClassifyDecisionTree(BaseDecisionTree):
    """分类决策树"""

    @classmethod
    def calc_gini(cls, d):
        """
        计算基尼指数
        :param d: 一维数据数组
        :return: 基尼指数
        """
        # 数组长度
        length = len(d)
        # 获取该一维数组中各个元素的集合和对应的计数值
        values, counts = np.unique(d, return_counts=True)
        # 计算基尼指数的公式是：
        # Gini(D) = \sum_{k=1}^{K}p_k(1-p_k) = 1 - \sum_{k=1}^{K}(p_k)^2
        # 初始基尼指数为1
        gini = 1
        # 遍历数据元素集合
        for value, count in zip(values, counts):
            gini -= np.power(count / length, 2)
        return gini

    @classmethod
    def calc_entropy(cls, d):
        """
        计算一维数组的熵
        :param d: 一维数据数组
        :return: 熵
        """
        # 数组长度
        length = len(d)
        # 获取该一维数组中各个元素的集合和对应的计数值
        values, counts = np.unique(d, return_counts=True)
        # 初始熵的值为0
        entropy = 0
        # 遍历每一个元素和其对应值以计算各部分的熵值
        # 熵的公式是：-\sum_{i=1}^{n}p_i\log{p_i}
        for value, count in zip(values, counts):
            entropy -= (count / length) * np.log2((count / length))

        return entropy

    @classmethod
    def calc_information_gain(cls, y, y1, y2):
        """
        计算每一个维度的特征的信息增益
        :param y: 标签集
        :param y1: 标签子集
        :param y2: 标签子集
        :return: 信息增益
        """
        # 计算整体数据集的熵值
        entropy = cls.calc_entropy(y)
        # 计算标签子集的占比
        p = len(y1) / len(y)
        # 计算信息增益
        information_gain = entropy - (p * cls.calc_entropy(y1)
                                      + (1 - p) * cls.calc_entropy(y2))

        return information_gain

    @classmethod
    def calc_information_gain_ratio(cls, y, y1, y2):
        """
        计算每一个维度的特征的信息增益
        :param y: 标签集
        :param y1: 标签子集
        :param y2: 标签子集
        :return: 信息增益率
        """



class RegressionDecisionTree(BaseDecisionTree):
    """回归决策树"""
    pass


if __name__ == '__main__':
    a = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                  [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 1]])
    b = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    print(ClassifyDecisionTree.calc_information_gains(a, b))
    print(ClassifyDecisionTree.calc_information_gain_ratios(a, b))
