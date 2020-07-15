import time

import numpy as np
from datasets.minist.loader import load_minist


def calc_gini(y):
    """
    计算基尼指数
    :param y: 一维数据数组
    :return: 基尼指数
    """
    # 数组长度
    length = len(y)
    # 获取该一维数组中各个元素的集合和对应的计数值
    values, counts = np.unique(y, return_counts=True)
    # 计算基尼指数的公式是：
    # Gini(D) = \sum_{k=1}^{K}p_k(1-p_k) = 1 - \sum_{k=1}^{K}(p_k)^2
    # 初始基尼指数为1
    gini = 1
    # 遍历数据元素集合
    for value, count in zip(values, counts):
        gini -= np.power(count / length, 2)
    return gini


def calc_feature_gini(y, y1, y2):
    """
    计算在特征A和取值a条件下的基尼指数
    :param y: 标签集
    :param y1: 在特征A处取值等于a对应的标签子集
    :param y2: 在特征A处取值不等于a对应的标签子集
    :return: 特征基尼指数
    """
    # 计算公式如下：
    # \frac{\vert D_1 \vert}{\vert D \vert}Gini(D_1) +
    # \frac{\vert D_2 \vert}{\vert D \vert}Gini(D_2)
    p = len(y1) / len(y)
    gini = p * calc_gini(y1) + (1 - p) * calc_gini(y2)

    # 信息增益和信息增益比都是选择大的，而基尼指数是选择小的
    # 所以为了保持算法的一致性，使用1减去基尼指数作为最终的返回结果
    return 1 - gini


def calc_entropy(y):
    """
    计算一维数组的熵
    :param y: 一维数据数组
    :return: 熵
    """
    # 数组长度
    length = len(y)
    # 获取该一维数组中各个元素的集合和对应的计数值
    values, counts = np.unique(y, return_counts=True)
    # 初始熵的值为0
    entropy = 0
    # 遍历每一个元素和其对应值以计算各部分的熵值
    # 熵的公式是：-\sum_{i=1}^{n}p_i\log{p_i}
    for value, count in zip(values, counts):
        entropy -= (count / length) * np.log2((count / length))

    return entropy


def calc_feature_information_gain(y, y1, y2):
    """
     计算在特征A和取值a条件下的信息增益
    :param y: 标签集
    :param y1: 在特征A处取值等于a对应的标签子集
    :param y2: 在特征A处取值不等于a对应的标签子集
    :return: 信息增益
    """
    # 计算整体数据集的熵值
    entropy = calc_entropy(y)
    # 计算标签子集的占比
    p = len(y1) / len(y)
    # 计算信息增益
    information_gain = entropy - (p * calc_entropy(y1)
                                  + (1 - p) * calc_entropy(y2))

    return information_gain


def calc_feature_information_gain_ratio(y, y1, y2):
    """
     计算在特征A和取值a条件下的信息增益率
    :param y: 标签集
    :param y1: 在特征A处取值等于a对应的标签子集
    :param y2: 在特征A处取值不等于a对应的标签子集
    :return: 信息增益率
    """
    # 计算整体数据集的熵值
    entropy = calc_entropy(y)
    # 计算信息增益
    information_gain = calc_feature_information_gain(y, y1, y2)
    # 计算信息增益率
    information_gain_ratio = information_gain / entropy

    return information_gain_ratio


class DecisionTreeNode(object):
    """决策树节点"""

    def __init__(self, feature_idx, cut_point, class_, left, right, parent, depth):
        """
        :param feature_idx: 最优特征索引
        :param cut_point: 最优切分点
        :param class_: 类别标签
        :param left: 左子树
        :param right: 右子树
        :param parent: 父节点
        :param depth: 当前节点在树中的深度
        """
        self.feature_idx = feature_idx
        self.cut_point = cut_point
        self.class_ = class_
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = depth


class BaseDecisionTree(object):
    """决策树基类，不直接使用"""

    # 支持使用的评判准则，key是评判准则的字符串名，value是允许调用的方法对象
    SUPPORTED_CRITERION = {}

    def __init__(self,
                 criterion,
                 min_samples_split=2,
                 min_impurity_split=1e-6,
                 max_depth=256):
        """
        决策树基类初始化方法
        :param criterion: 使用的评判准则，如信息增益、信息增益比、基尼指数等
        :param min_samples_split: 调整拆分的最小的样本数
        :param min_impurity_split: 调整拆分的最小不纯度
        :param max_depth: 树生成的最大深度
        """
        self.criterion = self.SUPPORTED_CRITERION[criterion]
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
        self.root = DecisionTreeNode(feature_idx=None,
                                     cut_point=None,
                                     class_=None,
                                     left=None,
                                     right=None,
                                     parent=None,
                                     depth=0)
        self._build_tree(self.root, x, y)

    def _build_tree(self, node, x, y):
        """
        递归地构建决策树
        :param 决策树节点
        :param x: 特征集
        :param y: 标签集
        :return:
        """
        # 计算决策树节点的类别
        node.class_ = self.calc_class(y)

        # 获取特征个数和样本个数
        n_features, n_samples = np.shape(x)

        # 如果已经达到最大深度或者最小分割数量
        if node.depth >= self.max_depth or n_samples < self.min_samples_split:
            return

        # 切分特征下标
        cut_feature_idx = None
        # 切分点
        cut_point = None
        # 当前最大的不纯度
        current_impurity = 0
        # 在所有可能的特征中遍历寻找切分特征和切分点
        for feature_idx in range(n_features):
            # 得到第 feature_idx 个测试特征
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

        # 如果当前最大的不纯度大于调整拆分的最小不纯度，继续递归构造树节点
        if current_impurity > self.min_impurity_split:
            # 根据索引获取切分最优特征
            feature = x[cut_feature_idx, :]
            # 切分出左子节点的数据子集和标签子集
            left_index = np.where(feature == cut_point)
            left_x = x[:, left_index[0]]
            left_y = y[left_index]
            # 切分出右子节点的数据子集和标签子集
            right_index = np.where(feature != cut_point)
            right_x = x[:, right_index[0]]
            right_y = y[right_index]
            # 构建左右子树
            node.left = self._build_tree(left_x, left_y, node.depth+1)
            node.right = self._build_tree(right_x, right_y, node.depth+1)

    def predict(self, x):
        """
        使用决策树进行单个样本的预测
        :param x:
        :return:
        """
        pass

    def calc_class(self, y):
        """计算叶子节点的类别信息"""
        raise NotImplemented


class ClassifyDecisionTree(BaseDecisionTree):
    """分类决策树"""

    SUPPORTED_CRITERION = {
        "information_gain": calc_feature_information_gain,
        "information_gain_ratio": calc_feature_information_gain_ratio,
        "gini": calc_feature_gini,
    }

    def calc_class(self, y):
        """计算类别标签，这里使用数据集中出现最多的类别"""
        # 获取各个值的集合和其对应的出现次数
        values, value_counts = np.unique(y, return_counts=True)
        # 最终返回的类别
        class_ = None
        # 当前最大的出现次数，初始化为0
        max_count = 0
        for value, value_count in zip(values, value_counts):
            # 如果该值的出现次数大于当前最大的出现次数，更新
            if value_count > max_count:
                class_ = value
                max_count = value_count
        return class_


class RegressionDecisionTree(BaseDecisionTree):
    """回归决策树"""
    pass


if __name__ == '__main__':
    a = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                  [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 1]])
    b = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    A = a[0]
    clf = ClassifyDecisionTree('gini')
    clf.fit(a, b)
