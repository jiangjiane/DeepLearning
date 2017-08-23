# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
理论知识：卷积神经网络
https://www.zybuluo.com/hanbingtao/note/485480
"""

import numpy as np


# 卷积层初始化
class ConvLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_out_size(self.input_height, filter_height, zero_padding, stride)
        self.output_array = np.zeros((self.filter_number, self.output_height, self.filter_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    # 确定卷积层输出大小
    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size-filter_size+2*zero_padding)/stride+1  # (W-F+2P)/S+1

    def forward(self, input_array):
        '''
        计算卷积层的输出
        :param input_array:结果
        :return:
        '''
        pass


# 保存卷积层的参数以及梯度，并且实现用梯度下降法更新参数
class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))

    def get_weight(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


# 激活函数 forward backward
class ReluActivator(object):
    def forward(self, weighted_input):  # 前向计算
        return max(0, weighted_input)

    def backward(self, output):  # 导数
        return 1 if output > 0 else 0
