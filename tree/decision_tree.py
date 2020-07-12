import time

import numpy as np
from datasets.minist.loader import load_minist


class BaseCriterion(object):
    """评判准则基类"""
    pass


class Entropy(BaseCriterion):
    """分类任务的评判准则：熵，使用信息增益比"""
    pass


class Gini(BaseCriterion):
    """分类任务的评判准则：基尼指数"""
    pass


class MSE(BaseCriterion):
    """回归任务的评判准则：均方差"""
    pass


class BaseSplitter(object):
    """划分器基类"""
    pass


class DecisionTreeNode(object):
    """决策树节点"""
    pass


class BaseDecisionTree(object):
    """决策树基类，不直接使用"""
    pass


class ClassifyDecisionTree(BaseDecisionTree):
    """分类决策树"""
    pass


class RegressionDecisionTree(BaseDecisionTree):
    """回归决策树"""
    pass
