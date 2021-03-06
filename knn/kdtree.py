import time

import numpy as np
from datasets.minist.loader import load_minist


class KDNode(object):
    """kd树的节点"""

    def __init__(self, features, labels, depth, parent=None, left=None, right=None):
        """
        :param features: 节点包含的特征数据
        :param labels: 节点包含的标签数据
        :param depth: 节点的深度
        :param parent: 节点的父节点
        :param left: 节点的左孩子节点
        :param right: 节点的右孩子节点
        """
        self.features = features
        self.labels = labels
        self.depth = depth
        self.parent = parent
        self.left = left
        self.right = right
        self.feature = None  # 节点实例对应的特征
        self.label = None  # 节点实例对应的标签


class KDTree(object):
    """kd树"""

    def __init__(self):
        self.root = None  # 根节点
        self.k = 0  # 特征的数量

    def build(self, x, y):
        """
        构造kd树
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :param y: 标签数组
        :return:
        """
        print("Start building k-dimension tree...")
        start = time.process_time()

        # 获取特征的数量
        self.k = x.shape[0]
        # 构造根节点
        self.root = KDNode(x, y, 0)
        # 开始递归创建整棵kd树的节点
        self._recursion_create_node(self.root)

        end = time.process_time()
        print(f"Build k-dimension tree complete, cost time: {end - start}s")

    def predict(self, x, k):
        """
        预测输入的实例的类别
        :param x: 实例特征
        :param k: 近邻个数
        :return: 实例类别label
        """
        # 向下递归找出包含目标点x的叶子节点
        leaf_node = self._visit_down(self.root, x)
        # 记录近邻点的数组
        points = []
        # 缓存近邻点与实例之间的距离数组
        distances = []
        # 向上递归找出k个近邻点
        self._visit_up(x, leaf_node, k, points, distances)

        # 按多数表决的分类决策规则得出最后的分类标签
        # 这里需要考虑，不能纯粹按类别数量进行投票
        # 如果有两个以上类别票数一样时，如何处理？
        # 这里使用选取总距离更近的一个类别作为输出标签
        # 类别的投票计数器，key是类别，value是该类别的票数
        vote_counter = {}
        # 类别的总距离
        label_distances = {}
        for index in range(len(points)):
            point = points[index]
            distance = distances[index]
            # 增加该类别票数
            vote_counter.setdefault(point.label, 0)
            vote_counter[point.label] += 1
            # 增加该类别的总距离
            label_distances.setdefault(point.label, 0)
            label_distances[point.label] += distance
        labels = []  # 最大票数的类别数组
        max_vote = max(vote_counter.values())  # 最大票数
        # 遍历投票计数器，统计类别数组
        for key, value in vote_counter.items():
            if value == max_vote:
                labels.append(key)
        # 判断类别数组的数量是否大于1
        if len(labels) > 1:  # 如果大于1，表示存在多个类别票数一样，所以需要继续计算
            label = None  # 最终返回的类别
            label_distance = float('inf')  # 当前最短的类别总距离，初始化为无穷大
            # 遍历最大票数的类别数组
            for candidate_label in labels:
                # 如果当前候选类别的总距离小于当前最短的类别总距离
                if label_distances[candidate_label] < label_distance:
                    label = candidate_label  # 更新返回的类别
                    label_distance = label_distances[candidate_label]   # 更新当前最短的类别总距离
            return label
        else:  # 如果小于等于1，表示只有一个类别，直接返回即可
            return labels[-1]

    def test(self, x, y, k):
        """
        测试kd树
        :param x: 特征向量矩阵，每一个样本的特征向量按列排
        :param y: 标签数组
        :param k: 近邻个数
        :return: 准确率accuracy
        """
        err = 0
        for i in range(x.shape[1]):
            predicted = self.predict(x[:, i], k)
            if predicted != y[i]:
                err += 1
        return 1 - (err / x.shape[1])

    def _recursion_create_node(self, node):
        """
        递归地创建kd树节点
        :param node: kd树的节点
        :return:
        """
        # 节点的实例数量应该大于0
        if node.features.shape[1] <= 0:
            return

        # 按照公式 $l = j % k +1$ 求出当前节点应该按照哪一个特征计算
        # 说明：因为下标从0开始，所以这里不需要加1；j是节点的深度，k是特征的数量
        feature_index = node.depth % self.k

        # 按第l个特征的数值大小对样本数据进行排序
        # 不能直接使用 np.median(node.data[feature_index]) 去获取中位数的值，
        # 因为如果样本数量为偶数，中位数的值没有对应的样本，其次，如果所有样本在该特征上的值一样（例如0），
        # 也无法使用 [:, np.where(node.data[feature_index] <= median)[0]] 取筛选左右子集
        # 所以，这里正确的做法应该是先排序，计算出中位数对应样本所在的下标（取左） int(node.data.shape[1]/2)
        sort_index = np.argsort(node.features[feature_index])  # argsort() 返回的是数组值从小到大的索引值
        node.features = node.features[:, sort_index]  # 对特征数据进行排序
        node.labels = node.labels[sort_index]  # 对标签数据进行相同的排序，保持特征-标签对应的一致性
        median_index = int((node.features.shape[1] - 1) / 2)  # 中位数对应样本所在的下标(不严格中位数)

        # 设置节点保存的实例特征和标签
        node.feature = node.features[:, median_index]
        node.label = node.labels[median_index]

        # 根据中位数对应实例去划分左右数据集合
        left_features = node.features[:, :median_index]
        left_labels = node.labels[:median_index]
        right_features = node.features[:, median_index + 1:]
        right_labels = node.labels[median_index + 1:]

        # 递归地构造左右子树
        if left_features.shape[1] > 0:
            node.left = KDNode(left_features, left_labels, node.depth + 1, parent=node)
            self._recursion_create_node(node.left)
        if left_features.shape[1] > 0:
            node.right = KDNode(right_features, right_labels, node.depth + 1, parent=node)
            self._recursion_create_node(node.right)

    def _visit_down(self, node, x):
        """
        向下递归找出包含目标点x的叶子节点
        :param node: kd树节点
        :param x: 目标点x
        :return: 叶子节点leaf_node
        """
        # 如果没有左右孩子节点即表示为叶子节点
        if node.left is None and node.right is None:
            return node

        # 计算使用哪一个维度的特征进行比较
        feature_index = node.depth % self.k
        # 获取切分点的坐标（即特征值）
        cut_point_feature = node.feature[feature_index]

        # 若目标值x当前维度的坐标小于切分点的坐标，则移动到左子节点，否则移动到右子节点
        if x[feature_index] < cut_point_feature:
            leaf_node = self._visit_down(node.left, x)
        else:
            leaf_node = self._visit_down(node.right, x)

        return leaf_node

    def _visit_up(self, x, node, k, points, distances):
        """
        向上递归找出近邻点
        :param x, 预测实例的特征
        :param node: kd树节点
        :param k, 需要的近邻点个数
        :param points: 记录近邻点的数组
        :param distances: 预测实例与近邻点之间的距离数组
        :return:
        """
        # 判断当前节点是否为根节点
        if node.parent is None:  # 如果是根节点
            # 如果节点的父节点为空，表示已经到达根节点
            # 这个时候只要在自己身上更新最近邻节点，并且不需要再向上递归
            self._update_nearest(x, node, k, points, distances)
            return
        else:  # 如果不是根节点
            # 获取兄弟节点
            brother_node = node.parent.right if node == node.parent.left else node.parent.left
            # 在自己和兄弟节点身上都要更新最近邻节点
            self._update_nearest(x, node, k, points, distances)
            self._update_nearest(x, brother_node, k, points, distances)
            # 继续向上递归
            self._visit_up(x, node.parent, k, points, distances)

    def _update_nearest(self, x, node, k, points, distances):
        """
        更新最近邻节点
        :param x, 预测实例的特征
        :param node: kd树节点
        :param k, 需要的近邻点个数
        :param points: 记录近邻点的数组
        :param distances: 预测实例与近邻点之间的距离数组
        :return:
        """
        # 计算当前节点与预测实例之间的距离
        distance = self._cal_distance(x, node.feature)

        # 判断近邻点个数是否已经大于k
        if len(points) > k:  # 如果近邻点个数已经大于k
            # 获取与预测实例距离最远的节点下标
            index = distances.index(max(distances))
            # 如果当前节点与预测实例的距离小于最大距离，则淘汰最大距离所对应的下标
            if distance < distances[index]:
                points[index] = node
                distances[index] = distance
        else:  # 如果近邻点个数没有大于k，直接加入即可
            points.append(node)
            distances.append(distance)

    @staticmethod
    def _cal_distance(a, b):
        """
        计算两点之间的欧式距离
        :param a: 特征数据
        :param b: 特征数据
        :return: 欧式距离distance
        """
        distance = np.sqrt(np.sum(np.power(a - b, 2)))
        return distance


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
    clf.build(data[0], data[1])
    print("Accuracy on test set(k=3): ", clf.test(data[2], data[3], 3))
