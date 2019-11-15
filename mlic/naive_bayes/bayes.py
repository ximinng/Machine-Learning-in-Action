# -*- coding: utf-8 -*-
"""
   Description : naive bayes
                 贝叶斯公式
                 p(xy) = p(x|y)p(y) = p(y|x)p(x)
                 p(x|y) = p(y|x)p(x)/p(y)
   Author :      xxm
"""

import numpy as np


class BernoulliNB:
    """
    This implementation of Bernoulli naive bayes
    """

    def fit(self, X_train, y_train):
        """
        Train the bayes classifier based on training data.
        :return: self
        """
        self.X = X_train
        self.y = y_train
        self.classes = np.unique(y_train)  # set of class label
        self.parameters = {}

        for i, c in enumerate(self.classes):
            X_index_c = X_train[np.where(y_train == c)]
            X_index_c_mean = np.mean(X_index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_index_c, axis=0, keepdims=True)
            parameters = {
                "mean": X_index_c_mean,
                "var": X_index_c_var,
                "prior": X_index_c.shape[0] / X_train.shape[0]  # 先验概率
            }
            self.parameters['class' + str(c)] = parameters
            print('class' + str(c))
        return self

    def _pdf(self, X, classes):
        """
        一维高斯分布的概率密度函数
        :param X:
        :param classes:
        :return:
        """
        eps = 1e-4
        mean = self.parameters['class' + str(classes)]['mean']
        var = self.parameters['class' + str(classes)]['var']

        numerator = np.exp(-(X - mean) ** 2 / (2 * var + eps))
        denominator = np.sqrt(2 * np.pi * var + eps)

        # 朴素贝叶斯假设(每个特征之间相互独立)
        # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
        result = np.sum(
            np.log(numerator / denominator),
            axis=1,
            keepdims=True
        )
        return result.T

    def _predict(self, X):
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters['class' + str(y)]['prior'])
            posterior = self._pdf(X, y)
            prediction = prior + posterior
            output.append(prediction)
        return output

    def predict(self, X):
        output = self._predict(X)
        output = np.reshape(output, (self.classes.shape[0], X.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction


def load_data_set():
    """
    创建数据集,都是假的 fake data set
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    获取所有单词的集合
    :param data_set: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocab_set = set()  # create empty set
    for item in data_set:
        vocab_set = vocab_set | set(item)  # union of two set
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocab_list: 所有单词集合列表
    :param input_set: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    result = [0] * len(vocab_list)
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            # 这个后面应该注释掉，因为对你没什么用，这只是为了辅助调试的
            print('the word: {} is not in my vocabulary'.format(word))
            pass
    return result
