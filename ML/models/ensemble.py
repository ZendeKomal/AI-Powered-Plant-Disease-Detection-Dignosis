import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from models.base_model import create_base_model

def build_ensemble(num_classes=5):
  
  inputs, [custom_cnn, vgg, dense, resnet] = create_base_model()

  custom_features = GlobalAveragePooling2D()(custom_cnn)
  vgg_features = GlobalAveragePooling2D()(vgg)
  dense_features = GlobalAveragePooling2D()(dense)
  resnet_features = GlobalAveragePooling2D()(resnet)

  combined = concatenate([custom_features, vgg_features, dense_features, resnet_features])

  x = Dense(1024, activation='relu')(combined)
  x = Dropout(0.5)(x)
  x = Dense(512, activation='relu')(x)

  outputs = Dense(num_classes, activation='softmax')(x)

  return Model(inputs=inputs, outputs=outputs)