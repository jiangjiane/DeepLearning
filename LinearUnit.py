# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
理论知识：线性单元和梯度下降
https://www.zybuluo.com/hanbingtao/note/448086
"""

from Perceptron import Perceptron

# 定义激活函数
f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元, 设置输入参数的个数'''
        Perceptron.__init__(self, input_num, f)

"""
以下部分实现线性单元
"""
def get_training_dataset():
    '''
    假设5个人的收入数据
    :return: 训练数据
    '''
    # 构建训练数据
    input_vec = [[5], [3], [8], [1.4], [10.1]]  # 工作年限
    labels = [5500, 2300, 7600, 1800, 11400]  # 期望的输出列表，月薪，与输入一一对应
    return input_vec, labels

def train_linear_unit():
    '''
    训练线性单元
    :return: 训练好的线性单元
    '''
    lu = LinearUnit(1)  # 创建感知器，输入参数的特征数为1（工作年限）
    input_vec, labels = get_training_dataset()
    lu.train(input_vec, labels, 10, 0.01)  # 训练，迭代10次，学习效率为0.01
    return lu  # 返回训练好的单元

if __name__ == '__main__':
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    print linear_unit  # 查看权重
    # 测试
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])
