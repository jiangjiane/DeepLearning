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
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weight(), self.output_array[f], self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    :param input_array:原始数据
    :param zp:补零圈数
    :return:
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth, input_height+2*zp, input_width+2*zp))
            padded_array[:, zp: zp+input_height, zp: zp+input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height+2*zp, input_width+2*zp))
            padded_array[zp: zp+input_height, zp: zp+input_width] = input_array
            return padded_array

def conv(input_array, kernel_array, output_array, stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    :param input_array:输入数据
    :param kernel_array:相当于filter
    :param output_array:输出数据
    :param stride:步长
    :param bias:filter的bias
    :return:
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width, kernel_height, stride) * kernel_array
            ).sum() + bias

def get_patch(input_array, i, j, kernel_width, kernel_height, stride):
    '''
    计算卷积值
    :param input_array:
    :param i:
    :param j:
    :param kernel_width:
    :param kernel_height:
    :param stride:
    :return:
    '''
    pass

def element_wise_op(array, op):
    '''
    对numpy数组进行element wise操作
    :param array:
    :param op:
    :return:
    '''
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

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
