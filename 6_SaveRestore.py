# -*- coding: utf-8 -*-
# @File    : 6_SaveRestore.py
# @Time    : 2018/5/24 8:32
# @Author  : hyfine
# @Contact : foreverfruit@126.com
# @Desc    : 模型的保存和读取

import tensorflow as tf


def save():
    """
    模型保存
    :return:
    """
    # 创建两个随机tensor
    v1 = tf.Variable(tf.random_normal([2, 2]), dtype=tf.float32, name='v1')
    v2 = tf.Variable(tf.random_normal([3, 3]), dtype=tf.float32, name='v2')
    init = tf.global_variables_initializer()

    # session执行init，然后查看变量值，之后保存
    with tf.Session() as sess:
        sess.run(init)
        print('v1:', v1.eval())
        print('v2:', v2.eval())
        # 保存结果
        saver = tf.train.Saver()
        # 这里后缀名ckpt，表示checkpoint，这个可以任意
        save_path = saver.save(sess, 'model/test.ckpt')
        print('model has saved to', save_path)


def restore():
    """
    模型加载
    :return:
    """
    # 为了验证，这里设置0初始化
    v1 = tf.Variable(tf.zeros([2, 2]), dtype=tf.float32, name='v1')
    v2 = tf.Variable(tf.zeros([3, 3]), dtype=tf.float32, name='v2')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'model/test.ckpt')
        print("V1:", v1.eval())
        print("V2:", v2.eval())
        print("Model restored")


if __name__ == '__main__':
    # save()
    #restore()
    pass
