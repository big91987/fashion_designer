import random
import cv2
import numpy as np
import multiprocessing as mp
import time
import os
from  functools import reduce


# class DataLoader():
#     def __init__(self, file_path, img_shape=(224, 224)):
#         # 获取到了数据条数
#         self.f_lines = open(file_path, encoding='gb18030').readlines()
#
#         # 定义了label编号
#         self.labelDict = {'吊带连衣裙': 0, 'T恤': 1, '无袖外套-马甲': 2, '披肩-斗篷': 3, '打底-吊带-背心-抹胸': 4, \
#                           '羽绒服-棉服': 5, '大衣-风衣': 6, '皮毛短外套': 7, '连衣裙': 8, '皮夹克': 9, \
#                           '卫衣': 10, '连体裤': 11, '衬衫': 12, '针织套衫': 13, \
#                           '夹克': 14, '西服': 15, '半身裙': 16, '中长裤': 17, \
#                           '套衫': 18, '针织外套': 19, '牛仔外套': 20, '皮毛大衣': 21, \
#                           '皮大衣': 22, 'polo衫': 23}
#         # 获取到数据量大小
#         self.dataNum = len(self.f_lines)
#         # 获取到label个数
#
#         self.labelNum = len(self.labelDict)
#         self.img_shape = img_shape
#
#     def load_batch(self, batch_size, is_training=True):
#         """
#         用于加载数据,batch_size为训练批次图像数目,is_training为是否训练标志
#         """
#         # 计算一个epoch的batch次数
#         batchNum = len(self.f_lines) // batch_size
#
#         pool = mp.Pool()
#
#         while True:
#
#             # 每个epoch之前做一次数据扰乱
#             random.shuffle(self.f_lines)
#             for i in range(batchNum):
#
#                 # 是一个batch的内容
#                 trainContent = self.f_lines[i * batch_size:(i + 1) * batch_size]
#
#                 # 存储训练的图像和训练的label
#                 imgs = []
#                 labels = []
#
#                 # 进行数据加载,可以编写为多线程或者多进程加速(直接读取速度会非常慢,尤其是图像数据比较大时)
#                 for index, content in enumerate(trainContent):
#
#                     # 数据格式   imgPath,xmin,ymin,xmax,ymax,label
#                     imgPath, xmin, ymin, xmax, ymax, label = content.strip().strip().split(",")
#                     img = cv2.imread(imgPath)
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     img_h, img_w, _ = img.shape
#
#                     # 做图像数据提取,可以先预处理好,此处可以做数据增强,如翻转、加噪、随机裁剪
#                     xmin, ymin, xmax, ymax = int(float(xmin) * img_w), int(float(ymin) * img_h), \
#                                              int(float(xmax) * img_w), int(float(ymax) * img_w)
#                     cutImg = img[ymin:ymax, xmin:xmax]
#
#                     # 做随机数据增强,可以做多样的随机判断
#                     if is_training:
#                         img = img if np.random.random() < 0.5 else np.fliplr(img)
#
#                     # 将label进行对应
#                     label_index = self.labelDict[label]
#                     label = np.zeros(self.labelNum)
#                     label[label_index] = 1  # [1,0,0,.....]
#
#                     img = cv2.resize(img, self.img_shape)
#                     imgs.append(img)
#                     labels.append(label)
#
#                 yield np.array(imgs), np.array(labels)

class pDataLoader():
    def __init__(self, file_path, img_shape=(224, 224)):
        # 获取到了数据条数
        self.f_lines = open(file_path, encoding='utf-8').readlines()

        # 获取到数据量大小
        self.data_num= len(self.f_lines)

        tmp = self.init_dict()

        self.label_dict = {}
        self.num_dict = {}
        # print(list(zip(tmp.items())))

        for name, dict in tmp.items():
            self.label_dict[name] = dict['id']
            self.num_dict[name] = dict['total_num']
        print(self.label_dict)
        print(self.num_dict)
        # 获取到label个数

        self.label_num = len(self.label_dict)
        self.img_shape = img_shape

    def init_dict(self, sort=False):
        # name_list = []
        obj_dict = {}
        for line in self.f_lines:
            line = line.strip()
            name = line.split(',')[-1]

            if name in obj_dict.keys():
                obj_dict[name]['total_num'] += 1
            else:
                obj_dict[name] = {
                    'name': name,
                    'id': len(obj_dict.keys()),
                    'total_num': 1,
                }

        if sort:
            for i, obj in enumerate(sorted([obj_dict[k] for k in obj_dict.keys()],key=lambda x:x['total_num'])):
                obj_dict[obj['name']]['id'] = i

        return obj_dict

    @classmethod
    def img_process(cls, data):
        # print('data == {}'.format(data))
        process_id = os.getpid()
        # for d in data:
        #     print('[pid: {}] d ====> {}'.format(process_id,d))

        time.sleep(random.randint(1, 5))
        print('[pid: {}] deal with data len is {}\n'.format(process_id, len(data)))
        return data


    def load_batch(self, batch_size, is_training=True, worker_num = 8):
        """
        用于加载数据,batch_size为训练批次图像数目,is_training为是否训练标志
        """
        # 计算一个epoch的batch次数
        batchNum = len(self.f_lines) // batch_size
        pool = mp.Pool(worker_num)
        while True:

            # 每个epoch之前做一次数据扰乱
            random.shuffle(self.f_lines)

            for i in range(batchNum):


                # 是一个batch的内容
                contents = self.f_lines[i * batch_size:(i + 1) * batch_size]

                # 存储训练的图像和训练的label
                imgs = []
                labels = []

                num_per_proc = batch_size // worker_num

                # 进行数据加载,可以编写为多线程或者多进程加速(直接读取速度会非常慢,尤其是图像数据比较大时)
                # for index, content in enumerate(contents):

                tmp = []
                for k in range(worker_num):
                    ans = pool.apply_async(pDataLoader.img_process,
                                           args=(contents[k * num_per_proc:(k + 1) * num_per_proc],))
                    tmp.append(ans)

                # pool.join()

                # ret = reduce(lambda x, y:
                #              x if isinstance(x, list) else x.get() + y if isinstance(y, list) else y.get(),
                #              tmp)

                ret = []
                for t in tmp:
                    tt = t.get()
                    ret += tt

                print('all worker done ... batch {}'.format(i))

                yield ret

        pool.close()
        pool.join()

                # yield np.array(imgs), np.array(labels)





    # def load_batch(self, batch_size, is_training=True):
    #     """
    #     用于加载数据,batch_size为训练批次图像数目,is_training为是否训练标志
    #     """
    #     # 计算一个epoch的batch次数
    #     batchNum = len(self.f_lines) // batch_size
    #
    #     while True:
    #
    #         # 每个epoch之前做一次数据扰乱
    #         random.shuffle(self.f_lines)
    #         for i in range(batchNum):
    #
    #             # 是一个batch的内容
    #             trainContent = self.f_lines[i * batch_size:(i + 1) * batch_size]
    #
    #             # 存储训练的图像和训练的label
    #             imgs = []
    #             labels = []
    #
    #             # 进行数据加载,可以编写为多线程或者多进程加速(直接读取速度会非常慢,尤其是图像数据比较大时)
    #             for index, content in enumerate(trainContent):
    #
    #                 # 数据格式   imgPath,xmin,ymin,xmax,ymax,label
    #                 imgPath, xmin, ymin, xmax, ymax, label = content.strip().strip().split(",")
    #                 img = cv2.imread(imgPath)
    #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                 img_h, img_w, _ = img.shape
    #
    #                 # 做图像数据提取,可以先预处理好,此处可以做数据增强,如翻转、加噪、随机裁剪
    #                 xmin, ymin, xmax, ymax = int(float(xmin) * img_w), int(float(ymin) * img_h), \
    #                                          int(float(xmax) * img_w), int(float(ymax) * img_w)
    #                 cutImg = img[ymin:ymax, xmin:xmax]
    #
    #                 # 做随机数据增强,可以做多样的随机判断
    #                 if is_training:
    #                     img = img if np.random.random() < 0.5 else np.fliplr(img)
    #
    #                 # 将label进行对应
    #                 label_index = self.labelDict[label]
    #                 label = np.zeros(self.labelNum)
    #                 label[label_index] = 1  # [1,0,0,.....]
    #
    #                 img = cv2.resize(img, self.img_shape)
    #                 imgs.append(img)
    #                 labels.append(label)
    #
    #             yield np.array(imgs), np.array(labels)

def test_pDataLoader():
    p = pDataLoader(
        file_path='20190501.txt'
    )
    # p.init_dict()
    gen = p.load_batch(64)

    for kkkk in gen:
        print('kkkk is {}, \nlen is {} '.format(kkkk, len(kkkk)))

if __name__=='__main__':
    test_pDataLoader()