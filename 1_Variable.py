import tensorflow as tf

# 变量需要通过tensorflow的Variable转换成tensor的变量类型
# 此处需要注意a和b的行列，必须符合向量乘法的定义
a = tf.Variable([[0.5, 1.0]])
b = tf.Variable([[2.0], [1.0]])
x = tf.matmul(a, b)
print(a)
print(b)
# eval是evaluation的缩写，表示其真实的内容值。
# print(x.eval())

# 初始化variable，理解为将内容填入容器的过程的操作
init_op = tf.global_variables_initializer()
# tensor的操作必须通过一个session执行
with tf.Session() as sess:
    sess.run(init_op)
    print(x.eval())

# 常用操作
# 创建0矩阵，注意数据类型通常用float32，避免各种莫名其妙的错误
c = tf.zeros([3,4],dtype=tf.float32)
# 创建一个和b同形的0矩阵
d = tf.zeros_like(b)
# 元素为1的矩阵，ones_like同理
e = tf.ones([3,4],tf.int32)
# 常量
tensor = tf.constant([1,2,3,4,5])
tensor = tf.constant(-1.0, shape=[2, 3])
# 生成序列
tf.linspace(10.0, 12.0, 3, name="linspace")
r = tf.range(0, 10, 2)
