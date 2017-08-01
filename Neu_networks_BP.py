# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
理论知识：神经网络和反向传播算法
https://www.zybuluo.com/hanbingtao/note/476663
"""

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
        self.output = output
    
