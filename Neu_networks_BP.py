# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
理论知识：神经网络和反向传播算法
https://www.zybuluo.com/hanbingtao/note/476663
"""
import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


# node类, 负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算
class Node(object):
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象
        :param layer_index:节点所属层的编号
        :param node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        设置节点的输出值.如果节点属于输入层将会用到这个函数
        :param output: 需要设置的值
        :return:
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        添加一个下游节点的连接
        :param conn:
        :return:
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        添加一个到上游节点的连接
        :param conn:
        :return:
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        计算节点的输出
        :return:
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def cale_hidden_layer_delta(self):
        '''
        计算隐藏层的delta
        :return:
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        输出层的delta
        :param label:实际值
        :return:
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        打印节点的信息
        :return:
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' +str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


# ConstNode类，为了输出恒为1的节点，即偏置项Wb
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象
        :param layer_index: 节点所属层的编号
        :param node_index: 节点的编号
        :return:
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0

    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        :param conn:
        :return:
        '''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        计算隐藏层的delta
        :return:
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        打印节点信息
        :return:
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


# Layer类，负责初始化一层。另外，作为Node的集合对象，提供对Node集合的操作
class Layer(object):
    def __init__(self, layer_index, node_count):
        '''
        初始化一层
        :param layer_index: 层编号
        :param node_count: 层所包含的节点个数
        :return:
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        设置层输出，当为输入层时用到
        :param data:
        :return:
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        计算层的输出向量
        :return:
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        打印层信息
        :return:
        '''
        for node in self.nodes:
            print node


# Connection类，主要记录连接的权重，以及这个连接所关联的上下游节点
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        初始化连接，权重初始化为一个很小的随机数
        :param upstream_node: 连接的上游节点
        :param downstream_node: 连接的下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = np.random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        计算梯度
        :return:
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        '''
        获取当前的梯度
        :return:
        '''
        return self.gradient

    def update_weight(self, rate):
        '''
        根据梯度下降算法更新权重
        :param rate:
        :return:
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        打印连接信息
        :return:
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (self.upstream_node.layer_index,
                                            self.upstream_node.node_index,
                                            self.downstream_node.layer_index,
                                            self.downstream_node.node_index,
                                            self.weight)


# Connections类，提供Connection集合操作
class Connections(object):
    def __init__(self):
        '''
        连接集合
        '''
        self.connections = []

    def add_connection(self, connection):
        '''
        添加连接集合
        :param connection:
        :return:
        '''
        self.connections.append(connection)

    def dump(self):
        '''
        打印连接信息
        :return:
        '''
        for conn in self.connections:
            print conn


# Network类，提供API
class Network(object):
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络
        :param layers: 二维数组，描述神经网络每层节点数
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        print layer_count
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connections(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        '''
        训练神经网络
        :param labels:数组，训练样本标签；每个元素都是一个样本的标签
        :param data_set: 二维数组，训练样本特征；每个元素都是一个样本的特征
        :param rate: 学习效率
        :param iteration: 迭代次数
        :return:
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        内部函数，用一个样本训练网络
        :param label:
        :param sample:
        :param rate:
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        :param sample: 数组，样本的特征，也就是网络的输入向量
        :return:
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def calc_delta(self, label):
        '''
        内部函数，计算每个节点的delta
        :param label:
        :return:
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        内部函数，更新每个连接权重
        :param rate:
        :return:
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        :return:
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        获取在一个样本下网络上每个连接上的梯度
        :param label: 样本标签
        :param sample: 样本输入
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def dump(self):
        '''
        打印网络信息
        :return:
        '''
        for layer in self.layers:
            layer.dump()


# 检查神经网络是否写错，通过梯度检查
def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    :param network: 神经网络对象
    :param sample_feature: 样本特征
    :param sample_label: 样本标签
    :return:
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式子计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (expected_gradient, actual_gradient)
