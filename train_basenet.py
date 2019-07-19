from utils.model_utils import *
import cv2
import numpy as np
from os import path

# 如果用import 要写成import utils\,调用的时候写utils.basenet

BATCH_SIZE = 1
HEIGHT = 224
WIDTH = 224


def fetch_batch():
    # 这里路径名要重新写一下，改成os.path
    # path="car_example.png"，如果是和当前文件在同一个文件夹下的文件就可以这么写
    # path="data\car_example.png"如果和是在当前文档同文件夹的子文件夹下，可以这么写，但注意最开始没有\
    # i_path="C:\Dream\codeworkspace\DensePCByImageFeature\data\car_example.png"
    directory = path.dirname(__file__)  # file表示当前文件路径包括文件名、所以要获取所在目录要用dirname
    img_path = path.join(directory, "data", "car_example.png")  # os.path.join("","","")用于拼接.从第一个\开头的开始拼接，所以不要随便加\
    print(img_path)
    ip_image = cv2.imread(img_path)
    ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)  # cv2默认读取是BGR，所以要换一下
    ip_image = np.array(ip_image)  # 转为n维数组
    # print(type(ip_image))  #type(a)是a的类型，a.dtype是a中的数据的类型
    '''
    ip_image = tf.expand_dims(ip_image, 0) 这里如果这么做会导致返回的ip_image变为tensor类型，从而
    导致feedict的时候报错，因为feeddict里面的用的place_holder所以最好都使用Numpy里的格式转换方式
    '''
    ip_image = np.reshape(ip_image, [BATCH_SIZE, HEIGHT, WIDTH, 3])  # 在0处增加一个维度，变为[batchsize,224,224,3]
    # print(type(ip_image))
    return ip_image


def test_basenet():
    """
        placeholder占位符，用于得到在运行时传递进来的真实的训练样本
        不像tf.Variable，不用指定初始值，但要说明类型，shape。可以在运行时，通过session.run的函数的
        feed_dict参数指定
        """
    img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')

    # 构建graph
    with tf.variable_scope('basenet_vars'):
        out_base = basenet(img_inp)

    # GPU设置,用在创建session的时候，用来对session进行参数配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    loss = out_base
    # run session 退出session的时候释放资源
    with tf.Session(config=config) as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        x = fetch_batch()
        pc = sess.run(loss, feed_dict={img_inp: x})

        print(pc[0, 6:10, :])  # 其结果的确是在[-1,1]之间的


def extract_resnet_features(img_input, model_path):
    """

    :param img_input: bn*224*224*3的图
    :param model_path: resnet50模型所在文件夹
    :return: 输入图片在resnet的输出,大小为BS*7*7*2018
    """

    graph = tf_utils.load_frozen_model(model_path, print_nodes=False)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=sess_config)

    feed_dict = {"input:0": img_input}
    features = sess.run("res5c:0", feed_dict=feed_dict)
    sess.close()  # 设置回原来的graph
    tf.reset_default_graph()
    return features


if __name__ == '__main__':
    # test_basenet()
    x = fetch_batch()

    # 获取renet50模型的路径
    directory = os.path.dirname(__file__)  # os.path.direname可以获得参数（文件/目录）的父级目录
    model_path = os.path.join(directory, "pretrained_model", "reg.pb")

    # test regional feature的获取
    features = extract_resnet_features(x, model_path)
    print(features.shape)
