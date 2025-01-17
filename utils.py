from random import shuffle
import cv2
import numpy as np
from PIL import Image
import os
def get_file_name(path):
    """
    return filename
    such as :[100.png,101.png]->[100,101]
    """
    filename = os.listdir(path)
    filename = [file.split(".")[0] for file in filename]
    return np.array(filename)

def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a

def letterbox_image(image, label, size):
    label = Image.fromarray(np.array(label))

    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    label = label.resize((nw, nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
    return new_image, new_label


class Generator(object):
    def __init__(self, batch_size, train_lines, image_size, num_classes):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")

        # flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label

    def generate(self, random_data=True):
        """
        新的网络数据集生成时候需要做到：
        1. 训练集图片的尺寸为256*256*3
        2. 训练集标签生成256*256*8
        """
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            # 从文件中读取图像
            jpg = Image.open("./Data/train/train_image" + '/' + name + ".png")
            png = Image.open("./Data/train/train_label" + '/' + name + ".png")

            if random_data:
                jpg, png = self.get_random_data(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))
            else:
                jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))

            inputs.append(np.array(jpg) / 255)

            # 从文件中读取图像
            png = np.array(png)
            png[png > 0 ] = 1

            # 转化成256*256*8的形式
            png = np.array([png] * 8)
            seg_labels = np.einsum('ijk->jki', png)
            targets.append(seg_labels)

            i = (i + 1) % length
            # 一个batch有两个
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets
