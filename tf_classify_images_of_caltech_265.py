import os

import matplotlib.pyplot as plt
import tensorflow as tf

from tf_function import load_and_preprocess_image

"""
@author vinson
@date 2020年4月15日
@desc 使用Caltech_265训练图像分类神经网络
@reference http://www.vision.caltech.edu/Image_Datasets/Caltech256/

此图片数据的shape都不相同
"""

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 加载和预处理数据集
path = "D:\\vinson\\Desktop\\飞浆\\data_set\\image_\\256_ObjectCategories"
# 标签
classes = os.listdir(path)
# 标签索引dict
label_index = dict((name, index) for index, name in enumerate(classes))

images = []  # 存放图片路径
labels = []  # 存放标签索引

for c in classes:
    data_path = os.path.join(path, c)
    data = os.listdir(data_path)
    for d in data:
        images.append(os.path.join(data_path, d))
        labels.append(label_index[c])

# 加载和格式化图片

# 构建数据集
path_ds = tf.data.Dataset.from_tensor_slices(images)

image_ds = path_ds.map(load_and_preprocess_image)

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2, 2, n + 1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    break

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

# 由于这些数据集顺序相同，你可以将他们打包在一起得到一个(图片, 标签)对数据集：
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
