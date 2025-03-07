import tensorflow as tf
import json
import os

def load_dataset(dataset_path, img_size=(256, 256), batch_size=32, validation_split=0.2):
  class_names = sorted([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])

  with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

  train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset='training',
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
  )

  val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
  )

  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

  return train_ds, val_ds, class_names
