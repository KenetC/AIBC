import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Cropping2D
from keras.callbacks import EarlyStopping

import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' ||| export TF_ENABLE_ONEDNN_OPTS=0

device = tf.config.list_physical_devices('GPU')

if device:
    print("GPU disponible: ", device)
else:
    print("No se detectaron GPUs")

def diff(X, Y): 
    diffY = Y.shape[1] - X.shape[1]
    diffX = Y.shape[2] - X.shape[2]
    return diffY, diffX

def R(N): 
    '''
    N: shape[1] of input image
    R: shape[1] to concat up-conv with encoder 
    '''
    n1 = N-4; n2 = np.floor((n1-2)/2)-3; n3 = np.floor((n2-2) / 2)-3; n4 = np.floor((n3-2)/2)-3
    n5 = np.floor((n4-2)/2)-3
    
    u1 = n5 * 2 ; dif = int(n4 - u1)
    c1 = (dif // 2,  dif - dif // 2)
    u2 = (u1 - 4)*2; dif = int(n3 - u2)
    c2 = (dif // 2,  dif - dif // 2)
    u3 = (u2 - 4)*2; dif = int(n2 - u3)
    c3 = (dif // 2,  dif - dif // 2)
    u4 = (u3 - 4)*2; dif = int(n1 - u4) 
    c4 = (dif // 2,  dif - dif // 2)
    return c1,c2,c3,c4

class Conv_3(Model): 
    def __init__(self, out_channels):
        super().__init__() 
        self.conv = Conv2D(out_channels,kernel_size=3,activation="relu", kernel_initializer = "he_normal", padding="valid")
    def call(self, input): 
        return self.conv(input)

class Double_Conv(Model):
    def __init__(self, out_channels, p):
        super().__init__() 
        
        self.double_conv = keras.Sequential([
            Conv_3(out_channels),
            Dropout(p),
            Conv_3(out_channels)
        ])
    def call(self, input): 
        return self.double_conv(input)

class Down_Conv(Model):
    def __init__(self, out_channels, p):
        super().__init__() 
        
        self.encoder = keras.Sequential([
            MaxPooling2D(pool_size =2, strides=2),
            Double_Conv(out_channels,p)
        ])  
    def call(self, input): 
        return self.encoder(input)

class Up_Conv(Model): 
    def __init__(self, out_channels, p, c): 
        super().__init__()

        self.Trans_Conv = Conv2DTranspose(out_channels, kernel_size=2, strides=2, padding='valid')
        self.decoder = Double_Conv(out_channels,p)
        self.c1, self.c2= c
    
    def call(self, x1, x2, final:bool=False): 
        '''
        x1 : input to transpose conv
        x2 : Volume from down sample to concat
        '''
        # print("x1: ", x1.shape)
        # print("x2: ", x2.shape)
        x1 = self.Trans_Conv(x1)
        # crop_height, crop_width = diff(x1,x2)
        # print('crop_height ',crop_height // 2 , "---", crop_height - crop_height // 2)
        # print('crop_width ',crop_width // 2 , "---", crop_width - crop_width // 2)

        # q = Cropping2D(cropping = ((crop_height // 2, crop_height - crop_height // 2),(crop_width // 2, crop_width - crop_width // 2) ))(x2)
        q = Cropping2D(cropping = ((self.c1,self.c2),(self.c1,self.c2)) )(x2)
        x = Concatenate(axis = -1)([q,x1])
        if final : 
            x = Cropping2D(cropping = ((2,2),(2,2)))(x)
        x = self.decoder(x)
        return x

class Unet_model(Model): 
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, base:int=64, p:float=0.3): 
        super().__init__()
        '''
        IMG_HEIGHT == IMG_WIDTH and IMG_CHANNELS = 3 
        '''
        self.IMG_HEIGHT = IMG_HEIGHT; self.IMG_WIDTH = IMG_WIDTH; self.IMG_CHANNELS = IMG_CHANNELS
        a,b,c,d = R(self.IMG_WIDTH)

        self.first_conv = Double_Conv(base,p)
        self.down_conv1 = Down_Conv(2*base,p)
        self.down_conv2 = Down_Conv(4*base,p)
        self.down_conv3 = Down_Conv(8*base,p)

        self.Middle_conv = Down_Conv(16*base,p)

        self.up_conv1 = Up_Conv(8*base,p,a)
        self.up_conv2 = Up_Conv(4*base,p,b)
        self.up_conv3 = Up_Conv(2*base,p,c)
        self.up_conv4 = Up_Conv(base,p,d)

        self.final_conv = Conv2D(1,kernel_size=1, strides=1,padding="same", activation="sigmoid")
    
    def call(self, input):
        # print(input)
        # print("tipo ",type(input))
        # if input.shape[1:] != (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS):
        #     raise ValueError(f"Forma de entrada incorrecta. Se esperaba (X, {self.IMG_HEIGHT}, {self.IMG_WIDTH}, {self.IMG_CHANNELS}), pero se recibió {input.shape}")

        x1 = self.first_conv(input)
        x2 = self.down_conv1(x1)
        # print('dow_conv-1')
        # print(x2.shape)
        x3 = self.down_conv2(x2)
        # print('dow_conv-2')
        # print(x3.shape)
        x4 = self.down_conv3(x3)
        # print('dow_conv-3')
        # print(x4.shape)

        x5 = self.Middle_conv(x4)
        # print('dow_conv-4')
        # print(x5.shape)

        # print("entrando u1--------------")
        u1 = self.up_conv1(x5,x4)
        # print(u1.shape)
        # print("entrando u2--------------")
        u2 = self.up_conv2(u1,x3)
        # print(u2.shape)
        # print("entrando u3--------------")
        u3 = self.up_conv3(u2,x2)
        # print(u1.shape)
        # print("entrando u4--------------")
        u4 = self.up_conv4(u3,x1, final = True)

        x = self.final_conv(u4)
        return x 
