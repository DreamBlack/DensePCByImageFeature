import os
import tensorflow as tf


def load_frozen_model(pb_path, prefix='', print_nodes=False):
    """
    从pb文件中加载模型，重新恢复模型后，可以通过
    graph.get_tensor_by_name('prefix>/<op_name>')范围
    :param pb_path: 模型路径
    :param prefix: 加载操作之前的前缀
    :param print_nodes: 是否要输出n结点名
    :return: tensorflow graph 定义
    """
    if os.path.exists(pb_path):
        with tf.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name=prefix
            )
            if print_nodes:
                for op in graph.get_operations():
                    print(op.name)
            return graph
    else:
        print('Model file does not exist', pb_path)
        exit(-1)


#  测试写的对不对
if __name__ == '__main__':
    directory=os.path.dirname(__file__)  # os.path.direname可以获得参数（文件/目录）的父级目录
    directory=os.path.dirname(directory)
    pb_path=os.path.join(directory,"pretrained_model","reg.pb")

    load_frozen_model(pb_path,"reg",True)
