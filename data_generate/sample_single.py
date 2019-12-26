import numpy as np
import sklearn.preprocessing
from data_generate.obj_io import parse_obj_file
import os
def camera_info(param):
    '''

    :param param: 1*3，应该是相机的坐标？球面坐标系的三个角度
    :return: cam_pos是相机点在原坐标系里面，直角坐标系下的坐标
              cam_mat是新的世界下的原点在就坐标系下的坐标系
    '''
    theta = np.deg2rad(param[0])  #角度转弧度
    phi = np.deg2rad(param[1])

    # 化为直角坐标系坐标
    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    # 求三个轴 坐标
    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)  # 求叉积
    axisY = np.cross(axisZ, axisX)

    # 单位化
    cam_mat = np.array([axisX, axisY, axisZ]) # 3*3,每一列对应一个坐标轴的坐标
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)  # 单位化
    return cam_mat, cam_pos


def sample_triangle(v, n=None):
    '''
    已知三角形某两条边坐标，要从中采样n个点
    :param v: 三角形某两条边坐标
    :param n: 要采样的点的个数
    :return: 采样的n个点的坐标
    '''
    if hasattr(n, 'dtype'):  # 如果对象有该属性返回 True，否则返回 False。dtype=int32
        n = np.asscalar(n)  # 将向量X转换成标量，且向量X只能为一维含单个元素的向量，就是给n整型
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert(v.shape[-2] == 2)

    # size=(n,2)，2大概是因为在平面上采样？，n是要采样的点的个数
    a = np.random.uniform(size=size)  # 从一个均匀分布[low,high)中随机采样，默认的就是01分布
    #  按行求和那就是n个，再n个里面取和大于1的，就超出了三角形返回
    mask = np.sum(a, axis=-1) > 1  # 按行求和，0是列，1是行，倒数第一个就是行，就是按行求和

    #  将超出三角形范围的再映射回1内
    a[mask] *= -1
    a[mask] += 1

    a = np.expand_dims(a, axis=-1)  # 在相应的axis轴上扩展维度,此处是最后一个轴,就是(n,2)变成(n,2,1)

    # a.shape:(n,2,1) v.shape:(2,3) sum之前a*v为n*2*3，sum之后shape就变为n*3
    # 就是根据比例a来求比例为a的v下的坐标
    return np.sum(a*v, axis=-2)

def sample_faces(vertices, faces, n_total):
    if len(faces) == 0:
        raise ValueError('Cannot sample points from zero faces.')
    # print(faces.shape)  # (32600, 3)个面，这里存的是面上每个点对应的点的Index
    # print(vertices.shape) # 7286*3个点
    tris = vertices[faces]  # 获取每个三角网格的三个点的坐标
    # print(tris.shape)  # (32600, 3, 3) 共32600个面，每个面对应三个点，每个点有xyz三个坐标
    n_faces = len(faces)

    d0 = tris[..., 0:1, :]  # 一个点A的坐标
    ds = tris[..., 1:, :] - d0  # 第三个点C的坐标分别减去第一个点A、第二个点B的坐标，得到的是两条边的长 (32600, 2, 3)

    assert(ds.shape[1:] == (2, 3))  # 断言语句用来测试是否满足条件，不满足直接触发异常，退出运行。
    # 计算每个三角面的面积，**2+开根号是求向量的标量长，也就是三角形的面积
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1])**2, axis=-1))  # 任意2边向量的叉积的绝对值的1/2即为三角形的面积

    cum_area = np.cumsum(areas)  # 按照所给定的轴参数返回元素的梯形累计和，这里没有参数就按一维数组累加，输入【1,2,3】会得到[1,3,6]

    cum_area *= (n_total / cum_area[-1])  # -1取最后一个元素，也就是这个物体的三角面的总面积，求的是根据面积大小计算每个面占模型的比例
    fac_num = np.round(cum_area).astype(np.int32)  # 取整后就表示需要在每个面采样多少个点

    positions = []
    last = 0
    for i in range(n_faces):  # 对于每个面i，fac_num个点
        n = fac_num[i] - last  # last是剩下的
        last = fac_num[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))  # 已知三角形某两条边坐标，要从中采样n个点

    # print(positions[3000:3010])  # positions是一个列表，每个列表中元素个数不同,所以列表中元素加起来的个数等于采样的点的个数
    # print(np.concatenate(positions, axis=0))  # concatenate将列表中各个元素按行连城一个新的array,
    return np.concatenate(positions, axis=0)  # np.concatenate((a,b,c...),axis)按轴将abc...连乘一个新的array

def sample_single(obj_path, view_path, output_folder,num):
    '''
    obj_path = '../datasets/ShapeNetCorev2/ShapeNetCore.v2/04530566/10155655850468db78d106ce0a280f87/models/model_normalized.obj'
    view_path = '../datasets/ShapeNet/ShapeNetRendering/04530566/10155655850468db78d106ce0a280f87/rendering/rendering_metadata.txt'
    output_folder = '../datasets/pointcloud/1024/'
    '''
    # 1 sampling
    with open(obj_path,'r') as f:
        vertices, faces = parse_obj_file(f)[:2]  # 获取顶点和面的信息，返回的是positions, face_positions(0,1项，不包括第2项)
    sample = sample_faces(vertices, faces, num)  # 从中采样16384个点

    # 2 tranform to camera view
    position = sample * 0.57  # 这里不知道为啥？？？

    cam_params = np.loadtxt(view_path)  # （24,5）

    for index, param in enumerate(cam_params):
        cam_mat, cam_pos = camera_info(param)
        pt_trans = np.dot(position - cam_pos, cam_mat.transpose())  # 每个点转换到新的坐标系下的值
        pt_trans = pt_trans.astype(np.float32)
        npy_path = os.path.join(output_folder, '{0:02d}.npy'.format(int(index)))  # 宽度为2的十进制补齐
        # np.save(npy_path, pt_trans)
        np.savetxt(npy_path.replace('npy','xyz'), pt_trans)

if __name__ == '__main__':

    '''
    绝对路径是以c:\d:开头（linux中以\开头）,
    相对路径是相对于工作目录也就是os.getcwd()所在的位置，.\demo.txt，
    表示的相对路径对应的绝对路径就是工作目录\demo.txt
    ..\表示所在目录的父目录
    .表示当前目录路径
    
    os.path提供了绝对路径和相对路径转换、检查给定路径是否为绝对路径等函数
    os.path.abspath(path)返回path参数的绝对路径字符串，可以将相对路径path，转为绝对路径
    os.path.relpath(path,start)返回从start改到path的相对路径的字符串
    os.path.dirname(path)将返回一个字符串，它包含 path 参数中最后一个斜杠之前的所有内容；
    os.path.basename(path) 将返回一个字符串，它包含 path 参数中最后一个斜杠之后的所有内容
    os.path.split(path) 获得path路径名中【目录名称，基本名称】组成的字符串元组
    '''

    obj_path = '../datasets/ShapeNetCorev2/ShapeNetCore.v2/04530566/' \
               '10155655850468db78d106ce0a280f87/models/model_normalized.obj'
    view_path = '../datasets/ShapeNet/ShapeNetRendering/04530566/' \
                '10155655850468db78d106ce0a280f87/rendering/rendering_metadata.txt'
    output_folder = '../datasets/pointcloud/1024/'
    sample_single(obj_path,view_path,output_folder)