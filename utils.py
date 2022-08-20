import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
import numpy as np
from variables import *

def preprocess_images(path):
    img = load_img(path, target_size=(IMG_NROWS, IMG_NCOLS))
    img = img_to_array(img)
    #Add 1 dimension to mark the number of batch
    img = np.expand_dims(img, axis=0)
    #Normalize array following VGG19 model
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_images(x):
    x = x.reshape((IMG_NROWS, IMG_NCOLS, 3))
    #BGR -> RGB
    x = x[:, :, ::-1]
    #Add mean value of normalization based on ImageNet
    x[:,:, 0] += 123.68 
    x[:,:, 1] += 116.779
    x[:,:, 2] += 103.939
    #Format value from 0 -> 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(content, generation):
    return tf.reduce_mean((generation - content)**2)

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, generation):
    G = gram_matrix(generation)
    A = gram_matrix(style)
    return tf.reduce_mean((G - A)**2)

# The learned synthesized image has a lot of high-frequency noise, using total variation denoising
# TVL = sum(i,j)[abs(xij - x(i+1)j) + abs(xij-xi(j+1))]

def total_variation_loss(x):
    return tf.image.total_variation(x)
