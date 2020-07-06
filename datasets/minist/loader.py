import os
import sys
import struct
import numpy as np

# 根路径
base = os.path.dirname(__file__)
# 训练集特征文件
train_images_file = os.path.join(base, 'data/train-images.idx3-ubyte')
# 训练集标签文件
train_labels_file = os.path.join(base, 'data/train-labels.idx1-ubyte')
# 测试集特征文件
test_images_file = os.path.join(base, 'data/t10k-images.idx3-ubyte')
# 测试集标签文件
test_labels_file = os.path.join(base, 'data/t10k-labels.idx1-ubyte')


def decode_idx3_ubyte(idx3_ubyte_file: str) -> np.ndarray:
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: data
    """
    # 读取二进制文件
    with open(idx3_ubyte_file, 'rb') as f:
        binary_data = f.read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片的宽和高
    offset = 0
    fmt_header = '>iiii'    # 因为数据结构中前4行数据类型都是32位整型，所有采用i格式。但是需要读取前4行，所以需要4个i
    magic_number, images_num, rows_num, cols_num = struct.unpack_from(fmt_header, binary_data, offset)

    # 解析数据集
    offset = struct.calcsize(fmt_header)    # 计算数据在缓存中的指针位置
    fmt_image = '>' + str(rows_num * cols_num) + 'B'    # 图像数据像素值类型为unsigned char,对应的format格式为B

    # 读取数据
    data = []
    for i in range(images_num):
        data.append(np.array(struct.unpack_from(fmt_image, binary_data, offset)))
        offset += struct.calcsize(fmt_image)

    return np.array(data)


def decode_idx1_ubyte(idx1_ubyte_file: str) -> np.ndarray:
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: data
    """
    # 读取二进制文件
    with open(idx1_ubyte_file, 'rb') as f:
        binary_data = f.read()

    # 解析文件头信息,依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, images_num = struct.unpack_from(fmt_header, binary_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    data = np.empty(images_num)
    for i in range(images_num):
        data[i] = struct.unpack_from(fmt_image, binary_data, offset)[0]
        offset += struct.calcsize(fmt_image)

    return data


def load_minist():
    """
    加载minist数据集
    :return: train_images, train_labels, test_images, test_labels
    """
    train_images = decode_idx3_ubyte(train_images_file)
    train_labels = decode_idx1_ubyte(train_labels_file)
    test_images = decode_idx3_ubyte(test_images_file)
    test_labels = decode_idx1_ubyte(test_labels_file)

    return [train_images, train_labels, test_images, test_labels]
