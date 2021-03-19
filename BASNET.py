from keras.layers import *
from Keras_BASNet.resnet34 import resnet_34
from keras.models import *

def ModelRefNet(image_inputs):
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(image_inputs)
    #256,256,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv1 = x
    x = MaxPooling2D(pool_size=(2,2))(x)

    #128,128,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #64,64,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #32,32,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #16,16,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #16 -> 32
    x = UpSampling2D(size=(2,2),interpolation="bilinear")(x)

    #32,32,64 + 32,32,64 -> 32,32,128
    x = Concatenate(axis=3)([x,conv4])
    #32,32,128 -> 32,32,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #32,32,64 -> 64,64,64
    ux = UpSampling2D(size=(2,2),interpolation="bilinear")(x)

    #64,64,64 + 64,64,64
    x = Concatenate(axis=3)([ux,conv3])
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 64,64,64 -> 128,128,64
    ux = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    # 128,128,64 + 128,128,64
    x = Concatenate(axis=3)([ux, conv2])
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 128,128,64 -> 256,256,64
    ux = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    # 256,256,64 + 256,256,64
    x = Concatenate(axis=3)([ux, conv1])
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    residual = Conv2D(filters=1, kernel_size=(3,3), padding='same')(x)
    ends = add([image_inputs,residual])

    return ends



def ModelBASNet(input_shape = (None,None,3)):
    inputs = Input(input_shape)
    feat1, feat2, feat3, feat4, feat5, feat6 = resnet_34(inputs)
    #bridge stage 8,8,512 -> 8,8,512
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same')(feat6)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #outputconvb
    outputconvb = Conv2D(filters=1, kernel_size=(3,3), padding='same')(x)
    outputconvb = UpSampling2D((32,32),interpolation="bilinear")(outputconvb)
    #decoder 1
    # 8,8,512 + 8,8,512 -> 8,8,1024
    x = Concatenate(axis=3)([feat6,x])
    # 8,8,1024 -> 8,8,512
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #outputconv6
    outputconv6 = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    outputconv6 = UpSampling2D((32, 32), interpolation="bilinear")(outputconv6)
    #8,8,512 -> 16,16,512
    x = UpSampling2D(size=(2,2),interpolation="bilinear")(x)

    #decoder 2
    #16,16,512 + 16,16,512 -> 16,16,1024
    x = Concatenate(axis=3)([feat5, x])
    # 16,16,1024 -> 16,16,512
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # outputconv5
    outputconv5 = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    outputconv5 = UpSampling2D((16, 16), interpolation="bilinear")(outputconv5)

    # 16,16,512 -> 32,32,512
    x = UpSampling2D(size=(2, 2),interpolation="bilinear")(x)

    # decoder 3
    # 32,32,512 + 32,32,512 -> 32,32,1024
    x = Concatenate(axis=3)([feat4, x])
    # 32,32,1024 -> 32,32,512
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # outputconv4
    outputconv4 = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    outputconv4 = UpSampling2D((8, 8), interpolation="bilinear")(outputconv4)
    # 32,32,512 -> 64,64,512
    x = UpSampling2D(size=(2, 2),interpolation="bilinear")(x)

    # decoder 4
    # 64,64,512 + 64,64,256 -> 64,64,768
    x = Concatenate(axis=3)([feat3, x])
    # 64,64,768 -> 64,64,256
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # outputconv3
    outputconv3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    outputconv3 = UpSampling2D((4, 4), interpolation="bilinear")(outputconv3)

    # 64,64,256 -> 128,128,256
    x = UpSampling2D(size=(2, 2),interpolation="bilinear")(x)

    # decoder 5
    # 128,128,256 + 128,128,128 -> 128,128,384
    x = Concatenate(axis=3)([feat2, x])
    # 128,128,384-> 128,128,128
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # 128,128,128 -> 256,256,128

    # outputconv2
    outputconv2 = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    outputconv2 = UpSampling2D((2, 2), interpolation="bilinear")(outputconv2)
    x = UpSampling2D(size=(2, 2),interpolation="bilinear")(x)

    # decoder 6
    # 256,256,128 + 256,256,64 -> 256,256,192
    x = Concatenate(axis=3)([feat1, x])
    # 256,256,192-> 256,256,64
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # outputconv1
    outputconv1 = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    dout = ModelRefNet(outputconv1)

    Final = Concatenate(axis=3)([dout,outputconv1,outputconv2,outputconv3,outputconv4,outputconv5,outputconv6,outputconvb])
    Final = Conv2D(filters=8,kernel_size=(1,1),strides=(1,1),padding='same',activation='sigmoid')(Final)

    model = Model(inputs = inputs, outputs = Final)

    return model






