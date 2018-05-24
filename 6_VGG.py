# -*- coding: utf-8 -*-
# @File    : 6_VGG.py
# @Time    : 2018/5/24 9:08
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : 加载预训练好的VGG模型参数，进行MINIST数据集分类任务
# 模型路径 ：model/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc

# ----------加载模型--------------
def vgg_net(model_path,input):
    """
    加载模型
    :param model_path:vgg模型路径
    :param input: 数据数据
    :return:
    """
    # vgg定义好的网络结构
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    # 加载mat格式的模型参数
    parameters = scipy.io.loadmat(model_path)
    print(parameters)



if __name__ == '__main__':
    vgg_net('model/imagenet-vgg-verydeep-19.mat','')