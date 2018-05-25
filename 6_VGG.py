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
import os
import matplotlib.pyplot as plt


# ----------加载模型--------------
# 一系列构建网络的辅助函数
def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def vgg_net(model_path, input):
    """
    加载模型，一次input在网络中的流动
    :param model_path:vgg模型路径
    :param input: 数据数据
    :return:
    """
    # vgg定义好的网络结构，只取了前面的35个step的参数，即提取特征图的部分，后面池化、全连接和softmax没有定义。
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
    data = scipy.io.loadmat(model_path)
    # normalization
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    # class labels
    # lables = data['classes'][0][0][0][0]
    # names = data['classes'][0][0][1][0]
    # wights and bias
    weights = data['layers'][0]
    # construct net
    net = {}
    current = input
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # 参数顺序转换
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        # 保存该层处理的结果（也就是特征图）
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel, layers


print('---------- VGG ready --------------')

if __name__ == '__main__':
    image_path = 'data/dog.jpg'
    vgg_path = 'model/imagenet-vgg-verydeep-19.mat'

    input_image = imread(image_path)
    shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    with tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        nets, mean_pixel, all_layers = vgg_net(vgg_path, image)
        input_image_pre = np.array([preprocess(input_image, mean_pixel)])
        layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1', 'relu3_4')
        for i, layer in enumerate(layers):
            print("[%d/%d] %s" % (i + 1, len(layers), layer))
            features = nets[layer].eval(feed_dict={image: input_image_pre})

            print(" Type of 'features' is ", type(features))
            print(" Shape of 'features' is %s" % (features.shape,))

            plt.figure(i + 1, figsize=(10, 5))
            plt.matshow(features[0, :, :, 0], cmap='gray', fignum=i + 1)
            plt.title("" + layer)
            plt.colorbar()
            plt.show()
