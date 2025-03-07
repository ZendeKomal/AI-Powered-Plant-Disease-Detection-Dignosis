import tensorflow as tf
from tensorflow.keras.applications import VGG16, DenseNet121, ResNet50
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, MaxPooling2D


def create_cnn(input_tensor):
    x = tf.keras.layers.Rescaling(1.0 / 255)(input_tensor)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    return x

def create_base_model(input_shape=(256,256, 3)):
  
  inputs = tf.keras.Input(shape=input_shape)
  custom_cnn = create_cnn(inputs)

  vgg_prep = tf.keras.applications.vgg16.preprocess_input(inputs)
  vgg = VGG16(weights='imagenet' ,include_top=False)(vgg_prep)

  dense_prep = tf.keras.applications.densenet.preprocess_input(inputs)
  dense = DenseNet121(weights='imagenet', include_top=False)(dense_prep)

  resnet_prep = tf.keras.applications.resnet.preprocess_input(inputs)
  resnet = ResNet50(weights='imagenet', include_top=False)(resnet_prep)

  return inputs, [custom_cnn, vgg, dense, resnet]
