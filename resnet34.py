from keras.layers import *
import keras.backend as K
from keras.models import *

def res_block(inputs,  filters, kernel_size = (3,3),
              strides = (1,1), padding = "same", with_conv_short_cut = False):
    conv1 = Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv2D(filters = filters, kernel_size=kernel_size, padding=padding)(conv1)
    conv2 = BatchNormalization()(conv2)

    if with_conv_short_cut:
        inputs = Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        x = ReLU()(add([conv2, inputs]))
        return x
    else:
        x = ReLU()(add([conv2, inputs]))
        return x

def resnet_34(Inputs):
    #256,256,3 -> 256,256,64
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(Inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #resnet_layer1 block1 256,256,64 -> 256,256,64
    x = res_block(inputs=x, filters=64)
    x = res_block(inputs=x, filters=64)
    feat1 = res_block(inputs=x, filters=64)

    #resnet_layer2 block2 256,256,64 -> 128,128,128
    x = res_block(inputs=feat1, filters=128, strides=(2,2), with_conv_short_cut=True)
    x = res_block(inputs=x, filters=128)
    x = res_block(inputs=x, filters=128)
    feat2 = res_block(inputs=x, filters=128)

    # resnet_layer3 block3 128,128,128 -> 64,64,256
    x = res_block(inputs=feat2, filters=256, strides=(2, 2), with_conv_short_cut=True)
    x = res_block(inputs=x, filters=256)
    x = res_block(inputs=x, filters=256)
    x = res_block(inputs=x, filters=256)
    x = res_block(inputs=x, filters=256)
    feat3 = res_block(inputs=x, filters=256)

    # resnet_layer4 block4 64,64,256 -> 32,32,512
    x = res_block(inputs=feat3, filters=512, strides=(2, 2), with_conv_short_cut=True)
    x = res_block(inputs=x, filters=512)
    feat4 = res_block(inputs=x, filters=512)

    # resnet_layer5 block6 32,32,512 -> 16,16,512
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(feat4)
    x = res_block(inputs=x, filters=512)
    x = res_block(inputs=x, filters=512)
    feat5 = res_block(inputs=x, filters=512)

    # resnet_layer6 block6 16,16,512 -> 8,8,512
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(feat5)
    x = res_block(inputs=x, filters=512)
    x = res_block(inputs=x, filters=512)
    feat6 = res_block(inputs=x, filters=512)


    return feat1,feat2,feat3,feat4,feat5,feat6







