from data_generate.sample_single import sample_single
import data_generate.shapenet_taxonomy
import os
import  numpy as np
from filecmp import dircmp
from progress.bar import IncrementalBar

def get_path(cat_id,num):
    '''
    根据类别号获取该类别图像对应模型所在的文件路径
    :param cat_id: 类别号
    :param num:要采样的点的个数
    :return:
    '''
    obj_path=[]
    view_path=[]
    output_folder=[]

    obj_prefix=os.path.join('../datasets/ShapeNetCorev2/ShapeNetCore.v2/',cat_id)
    view_prefix=os.path.join('../datasets/ShapeNet/shapeNetRendering/',cat_id)
    output_prefix=os.path.join('../datasets/pointcloud/',str(num),cat_id)

    if not os.path.isdir(output_prefix):
        os.makedirs(output_prefix)

    # 进行目录比较，并返回两个目录都存在的子目录的信息
    for i in dircmp(obj_prefix,view_prefix).common_dirs:
        obj_path.append(os.path.join(obj_prefix,i,'models/model_normalized.obj'))
        view_path.append(os.path.join(view_prefix, i, 'rendering/rendering_metadata.txt'))
        output_folder.append(os.path.join(output_prefix,i)) #要有24个

    for i in output_folder:
        if not os.path.isdir(i):
            os.makedirs(i)

    return obj_path,view_path,output_folder


def sample_all(num):
    '''
    下面要考虑的是
    已知的图片有，一个模型24个角度下的图片，那就要对应24个视角的信息，
    这24个图对应的是一个初始模型，对于这个初始模型要再利用24个视角的信息生成24个xyz文件来对应那24个图
    一种方法是
        对于每个cat_id下的每一个物体，get_path只生成04530566/10155655850468db78d106ce0a280f87的目录
        在本函数里，先对生成模型，再对24个视角进行变化，生成24个模型，存储在04530566/10155655850468db78d106ce0a280f87/下
        要注意文件名称和图片的名称对应相等
    :param: num表示要采样多少个点1024,4096等
    :return: 
    '''
    #for cat,cat_id in data_generate.shapenet_taxonomy.shapenet_category_to_id.items():

    cat_id='04530566'
    obj_path,view_path,output_folder=get_path(cat_id,num)
    print(obj_path)
    print(view_path)
    print(output_folder)

        # cat_id这个种类下需要生成len(obj_path)个xyz模型
    print('Sampling %d pointclouds for cat %s' % (len(obj_path), cat_id))
    bar = IncrementalBar(max=len(obj_path))
    for i in range(len(obj_path)):
            # 对于类别cat_id要生成的每个模型，根据obj文件所在路径、视角文件所在路径进行采样，并将结果输出到out_folder文件所在路径
        bar.next()
        sample_single(obj_path[i], view_path[i], output_folder[i],num)  # 啊啊啊啊！这里view_path是个list，得传下标的要
    bar.finish()
    #print('All cats sampled!')

if __name__ == '__main__':
    #a = np.loadtxt('../datasets/ShapeNet/ShapeNetRendering/04530566/10155655850468db78d106ce0a280f87/rendering/rendering_metadata.txt')  # 最普通的loadtxt
    #print(a)
    sample_all(4096)