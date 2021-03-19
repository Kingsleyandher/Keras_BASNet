import warnings
warnings.filterwarnings('ignore')
import os
from random import shuffle
from keras.models import *
from keras.layers import *
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)

from BASNET import ModelBASNet
from metrics import muti_bce_loss_fusion, output_loss,dice_coef,compute_IOU,ssim
from utils import Generator,get_file_name

#set parameters
class setup_config():
    optimizer = "Adam"
    lr = 1e-4
    init_Epoch = 0
    nb_epoch = 15
    def __init__(self, image_shape=(256,256,3), valid_percent = 0.2, nb_class=2, batch_size=16, init='random', verbose=1，log_dir = "./Logs/"):
        self.image_shape = image_shape
        self.nb_class = nb_class
        self.batch_size = batch_size
        self.init = init
        self.verbose = verbose
        self.valid_percent = valid_percent
        self.log_dir = log_dir

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and "ids" not in a:
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
config = setup_config(image_shape=(256,256,3),
                      valid_percent = 0.2,
                      nb_class=2,
                      batch_size=2,
                      init='random',
                      verbose=1，
                      log_dir = "./Logs/")
config.display()


#set model path
log_dir = config.log_dir #存放模型

#get data
TrainImagePath = "./Data/train/train_image/"
trainFile = get_file_name(TrainImagePath) 
ind_list = [i for i in range(len(trainFile))]
shuffle(ind_list)
nb_valid = int(len(trainFile)*config.valid_percent)
val_lines = trainFile[ind_list[:nb_valid]] 
train_lines = trainFile[ind_list[nb_valid:]] 

# load model
model = ModelBASNet()

checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
tensorboard = TensorBoard(log_dir=log_dir)


# train model
if True:

    model.compile(optimizer=config.optimizer,
                  loss=muti_bce_loss_fusion,
                  metrics=["binary_crossentropy", output_loss, dice_coef, compute_IOU,ssim])
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines),
                                                                               config.batch_size))

    gen = Generator(config.batch_size, train_lines, config.image_shape, config.nb_class).generate()
    gen_val = Generator(config.batch_size, val_lines, config.image_shape, config.nb_class).generate(False)

    model.fit_generator(gen,steps_per_epoch=max(1, len(train_lines) // config.batch_size),
                        validation_data=gen_val, validation_steps=max(1, len(val_lines) // config.batch_size),
                        epochs=config.nb_epoch,initial_epoch=config.init_Epoch,
                        callbacks=[checkpoint_period, reduce_lr, tensorboard],shuffle=True)
