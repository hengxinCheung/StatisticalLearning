import time

import numpy as np
from datasets.minist.loader import load_minist


class KDNode(object):
    """kd树的节点"""

    def __init__(self, data, depth, parent=None, left=None, right=None):
        """
        :param data: 节点包含的数据
        :param depth: 节点的深度
        :param parent: 节点的父节点
        :param left: 节点的左孩子节点
        :param right: 节点的右孩子节点
        """
        self.data = data
        self.depth = depth
        self.parent = parent
        self.left = left
        self.right = right


class KDTree(object):
    """kd树"""

    def __init__(self):
        self.root = None  # 根节点
        self.k = 0  # 特征的数量

    def build(self, x: np.ndarray):
        """
        构造kd树
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :return:
        """
        print("Start building k-dimension tree...")
        start = time.process_time()

        # 获取特征的数量
        self.k = x.shape[0]
        # 构造根节点
        self.root = KDNode(x, 0)
        # 开始递归创建整棵kd树的节点
        self._recursion_create_node(self.root)

        end = time.process_time()
        print(f"Build k-dimension tree complete, cost time: {end-start}s")

    def _recursion_create_node(self, node: KDNode):
        """
        递归地创建kd树节点
        :param node: kd树的节点
        :return:
        """
        if node.data.shape[1] <= 1:
            return

        # 按照公式 $l = j % k +1$ 求出当前节点应该按照哪一个特征计算
        # 说明：因为下标从0开始，所以这里不需要加1；j是节点的深度，k是特征的数量
        feature_index = node.depth % self.k
        # 求出第l个特征的中位数
        median = np.median(node.data[feature_index])

        # 根据中位数划分左右数据集合
        left_data = node.data[:, np.where(node.data[feature_index] <= median)[0]]
        right_data = node.data[:, np.where(node.data[feature_index] > median)[0]]

        # 构造左右节点
        if left_data.shape[1] >= 2:
            node.left = KDNode(left_data, node.depth+1, node)
            self._recursion_create_node(node.left)
        if right_data.shape[1] >= 2:
            node.right = KDNode(right_data, node.depth+1, node)
            self._recursion_create_node(node.right)


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
    data = load_data()
    clf = KDTree()
    clf.build(data[2])
