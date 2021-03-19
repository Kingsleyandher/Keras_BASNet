import colorsys
import copy
import os
import numpy as np
from PIL import Image
import cv2
from BASNET import ModelBASNet

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
# --------------------------------------------#
class BASNet(object):
    _defaults = {
        "model_path": 'Logs/ep014-loss0.678-val_loss0.562.h5',
        "model_image_size": (256, 256, 3),
        "num_classes": 2,
        # --------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        # --------------------------------#
        "blend": True,
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.model = ModelBASNet(self.model_image_size)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

        if self.num_classes == 2:
            self.colors = [(255, 255, 255), (0, 0, 0)]
        elif self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                          for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # ---------------------------------------------------#
        #   进行不失真的resize，添加灰条，进行图像归一化
        # ---------------------------------------------------#
        img, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        img = np.asarray([np.array(img) / 255])

        # ---------------------------------------------------#
        #   图片传入网络进行预测 得到256,256,8 0-1
        # ---------------------------------------------------#
        pr = self.model.predict(img)[0]
        pr = pr[...,0].reshape([self.model_image_size[0], self.model_image_size[1]]) #0-1
        #反归一化
        #256,256
        pr = (pr - pr.min())/(pr.max() - pr.min()) * 255.
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
             int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]
        # ------------------------------------------------#
        #   扩充成(,,3)
        # ------------------------------------------------#
        seg_img = np.expand_dims(pr,axis=2).repeat(repeats=3,axis=2)
        print(seg_img.shape)
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h), Image.NEAREST)
        # ------------------------------------------------#
        #   将新图片和原图片混合
        # ------------------------------------------------#
        if self.blend:
            image = Image.blend(old_img, image, 0.7)

        return image



basNet = BASNet()
path = "./Data/test/test_image/"
filepng = os.listdir(path)
for png in filepng:
    image = Image.open(path + png)
    r_image = basNet.detect_image(image)
    r_image.save("./Data/predict/" + png)
