import cv2
import numpy as np
import pickle
import os


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def cifar10_to_images():
    tar_dir = '../../dataSet/cifar-10-python/cifar-10-batches-py/'  # 原始数据库目录
    train_root_dir = '../../cifa-10/train/'  # 图片保存目录
    test_root_dir = '../../cifa-10/test/'
    if not os.path.exists(train_root_dir):
        os.makedirs(train_root_dir)
    if not os.path.exists(test_root_dir):
        os.makedirs(test_root_dir)
    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    label_names = ["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"]
    for j in range(1, 6):
        dataName = tar_dir + "data_batch_" + str(j)
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")

        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            dst = label_names[Xtr['labels'][i]] +'/'
            if not os.path.exists(train_root_dir + dst ):
                os.makedirs(train_root_dir + dst )
            picName = train_root_dir + dst + str(
                i + (j - 1) * 10000) + '.jpg'  # class 是文件夹名  index 是序号
            cv2.imwrite(picName, img)
        print(dataName + " loaded.")

    print("test_batch is loading...")

    # 生成测试集图片
    testXtr = unpickle(tar_dir + "test_batch")
    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        dst = label_names[testXtr['labels'][i]] +'/'
        if not os.path.exists(test_root_dir + dst):
            os.makedirs(test_root_dir + dst)
        picName = test_root_dir + dst + str(i) + '.jpg'
        cv2.imwrite(picName, img)
    print("test_batch loaded.")
cifar10_to_images()
