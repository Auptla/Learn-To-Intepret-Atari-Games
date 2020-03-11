import numpy as np
import tensorflow as tf
import keras

vgg = keras.applications.vgg19.VGG19(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=1000)
print(vgg.summary())