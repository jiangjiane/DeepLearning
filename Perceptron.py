# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
理论知识：https://www.zybuluo.com/hanbingtao/note/433855
"""

class Perceptron(object):
    def __init__(self, input_num, actovator):
        '''
        初始化感知器，设置参数输入的个数，以及激活函数
        '''
        self.activator = actovator
        self.weights = [0.0 for i in range(input_num)]  # 权重向量初始为0
        self.bias = 0.0  # 偏置项初始化为0

    def __str__(self):
        '''
        打印学习到的权重，偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        :param input_vec: [x1,x2,x3,...]
        :return: sum[x1*w1,x2*w2,...]
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,
                       zip(input_vec, self.weights)),
                   0.0) + self.bias)

    def train(self, input_vec, labels, iteration, rate):
        '''
        输入训练数据：一组向量，与每个向量对应的label；训练次数以及学习率
        :param input_vec: [x1,x2,x3,...]
        :param labels: xi corresponding label
        :param iteration: counts of iteration
        :param rate: constant
        :return:
        '''
        for i in range(iteration):
            self._one_iteration(input_vec, labels, rate)

    def _one_iteration(self, input_vec, labels, rate):
        '''
        一次迭代，把所有训练数据都过一遍
        :param input_vec: [x1,x2,x3,...]
        :param labels: xi corresponding label
        :param rate: constant
        :return:
        '''
        # 把输入和输出打包在一起，成为样本列表[(input_vec, label),...]
        samples = zip(input_vec, labels)  # 每个训练样本是(input_vec, label)
        for (input_vec, label) in samples:  # 对每个样本，按照感知器规则更新权重
            output = self.predict(input_vec)  # 计算感知器当前权重下的输出
            self._update_weights(input_vec, output, label, rate)  # 更新权重

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights))
        self.bias += rate * delta

"""
以下部分利用这个感知器实现and函数
"""
def f(x):
    '''
    定义一个激活函数f
    :param x: prerequisite
    :return: result of x
    '''
    return 1 if x > 0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    :return: input_vecs labels
    '''
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]  # 输入向量列表
    labels = [1, 0, 0, 0]  # 期望输出的列表，与输入一一对应[1,1]->1,[0,0]->0,[1,0]->0,[0,1]->0
    return input_vecs, labels

def train_and_perceptron():
    '''
    使用and真值表训练感知器
    :return: trained perceptron p
    '''
    p = Perceptron(2, f)  # 创建感知器，输入参数个数为2（and函数是二元函数），激活函数f
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)  # 训练，迭代10轮，学习速率为0.1
    return p  # 返回训练好的感知器

if __name__ == '__main__':
    # 训练and感知器
    and_perceptron = train_and_perceptron()
    print and_perceptron  # 获得的权重（因为定义了__str__,所以可以打印对象）
    # 测试
    print '1 and 1 = %d' % and_perceptron.predict([1, 1])
    print '0 and 0 = %d' % and_perceptron.predict([0, 0])
    print '1 and 0 = %d' % and_perceptron.predict([1, 0])
    print '0 and 1 = %d' % and_perceptron.predict([0, 1])
