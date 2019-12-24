import tensorflow as tf
from tensorflow.contrib.slim import nets
import decimal
import keyword

slim = tf.contrib.slim


def function(a):
    """" say something here！
    """

    if a==1:
        print(1)
    elif a==2:
        print(2)
    else:
        print(3)
    pass


print(function.__doc__)  # 调用 doc

class FooClass(object):
    def __init__(self,nm='John Doe'):
        self.name=nm

    def showname(self):
        print("your name is "+self.name)
        print("my name is "+self.__class__.__name__)

if __name__ == '__main__':
    foo1=FooClass()
    foo1.showname()
    foo2=FooClass("John Smith")
    foo2.showname()
    print(keyword.iskeyword("else"))
    print(function.__doc__)
    print(5/2)
    print(5//2)
    print(5.0/2)
    print(5.0//2)
    print(3>4or 6>5)
    print(--1)
    a=1.1
    print(type(foo2))
    print(str(foo2))
    print(repr(foo2))

    print((decimal.Decimal('1.1')))
    '''
    name
    '''
    for items in ['a', 'b', 'c']:
        print(items, end='')
    a=1
    function(a)
    # 只需要ckpt文件即可，不需要什么model文件了
    checkpoint_path = ""  # 训练好的权重加载的文件所在地，ckpt文件
    checkpoint_exclude_scopes = ""  # 那些不需要的那些遍历就不会被从checkpoint文件加载
    trainable_scopes = ""  # 如果只选训练其中一部分层，用这个来指定要训练的层们，剩下的会被frozen

    # 获取你需要restore使用的变量，可以用include，exclude去指定
    # scopes_to_freeeze="" #需要冻结的层
    # init_fn=get_init_fn(checkpoint_exclude_scopes)
    # vars_to_train=get_trainable_variables(scopes_to_freeeze)
