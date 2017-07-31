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
        pass
