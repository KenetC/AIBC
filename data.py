# import tensorflow as tf 
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage import exposure

import matplotlib.pyplot as plt

# from model import Unet_model

seed = 42
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'
train_ids = next(os.walk(TRAIN_PATH))[1]

dir_img = os.getcwd() + "/dataset/img"
dir_mask = os.getcwd() + "/dataset/mask"

def data_agu(dir_img,dir_mask,PATH): 
    train_ids = next(os.walk(PATH))[1]
    number = 0
    for i,id_ in enumerate(train_ids) :
        # if i == 100 : break
        # print(f" number: {number} : i: {i}")
        path = PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:3]
        N,M,_ = img.shape
        mask = np.zeros((N,M), dtype=np.uint8)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask = np.maximum(mask, mask_)
        n = N // IMG_WIDTH; m = M // IMG_HEIGHT
        while n > 0 : # filas
            if m == 0 : m = M // IMG_HEIGHT
            while m > 0 : # columnas
                imsave(dir_img + f'/img_{str(number).zfill(3)}.png', img[(n-1)*IMG_WIDTH:n*IMG_WIDTH,(m-1)*IMG_HEIGHT:m*IMG_HEIGHT]) 
                imsave(dir_mask + f'/mask_{str(number).zfill(3)}.png', mask[(n-1)*IMG_WIDTH:n*IMG_WIDTH,(m-1)*IMG_HEIGHT:m*IMG_HEIGHT])
                m -= 1; number += 1
            n -= 1


# config model
N = IMG_WIDTH + 2*94
m = Unet_model(N, N, 3, base = 10)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Ajusta la tasa de aprendizaje

m.compile(optimizer=optimizer, loss= tf.keras.losses.Dice(), metrics=['accuracy',tf.keras.metrics.BinaryIoU()])

## train 

# Obtener listas de archivos
image_files = sorted([os.path.join(dir_img, f) for f in os.listdir(dir_img)])
mask_files = sorted([os.path.join(dir_mask, f) for f in os.listdir(dir_mask)])
padds = [[94, 94], [94, 94], [0, 0]]


# image_files = image_files[:100]; mask_files = mask_files[:100]


# Aplicar skimage.exposure.equalize_adapthist usando tf.py_function
def equalize_adapthist(img, c:float= 0.005):
    '''
    img : type tensorflow.image
    '''
    img_np = (img.numpy()).astype(np.uint8) 
    equalized_img = exposure.equalize_adapthist(img_np, clip_limit=c)
    return equalized_img.astype(np.float32) 

# Función para cargar y preprocesar una imagen y máscara
def load_and_preprocess(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  
    
    image = tf.py_function(equalize_adapthist, inp=[image], Tout=tf.float32)
    # image = tf.cast(image, tf.float32) / 255.0
    image = tf.pad(image, paddings = padds, mode = "REFLECT")
    image.set_shape([N, N, 3])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0  

    return image, mask

# Crear el dataset
dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
dataset = dataset.map(load_and_preprocess)






# Configurar el dataset para el entrenamiento
BATCH_SIZE = 2
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=len(image_files))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Entrenar el modelo
EPOCHS = 10

validation_split = 0.2
# print("size image_f, ",len(image_files))
validation_size = int(len(image_files) * validation_split)
# print("tamanio de validation: ", validation_size)
train_dataset = dataset.skip(validation_size // BATCH_SIZE)
# print("tamanio dataset train: ",train_dataset.cardinality())
validation_dataset = dataset.take(validation_size // BATCH_SIZE)
# print("tamanio dataset val: ",validation_dataset.cardinality())
# early_stopping = EarlyStopping(patience=10, restore_best_weights=True)  , callbacks=[early_stopping]


history = m.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

m.save_weights(os.getcwd()+'/Nmodel.weights.h5')

# for inputs, targets in train_dataset:
#     print("Forma de inputs:", inputs.shape)
#     # print(inputs.shape[1])
#     print("Forma de targets:", targets.shape)
#     print("Tipo de inputs:", type(inputs))
#     print("Tipo de targets:", type(targets))
#     pre = m.predict(inputs)
#     print('forma salida: ', pre.shape)
#     break 

#     image = tf.keras.ops.expand_dims(image[0], axis=0)
#     prediction = m.predict(image)
#     print("predict shape: ")
#     print(prediction.shape)

# Crear una imagen de prueba
# img_height, img_width, img_channels = 440, 440, 3
# test_image = np.random.rand(img_height, img_width, img_channels).astype(np.float32)
# image = tf.expand_dims(test_image, axis=0) # Añadir la dimension del batch


# predict = m.predict(image)
# print("shape salida: ", predict.shape)
