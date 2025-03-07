import cv2
import numpy as np
import tensorflow as tf


class XAIProcess:
    def __init__(self, model):
        self.model = model
        self.custom_cam_layer = "conv2d_2"
        # self.pretrained_cam_layer = "vgg16"

    def multi_model_gradcam(self, img_array):
        if len(img_array.shape) == 3:
            img_array = tf.expand_dims(img_array, axis=0)

        custom_cam = self._grad_cam(img_array, self.custom_cam_layer)

        # pretrained_cam = self._grad_cam(img_array, self.pretrained_cam_layer)
        # combined_heatmap = np.maximum(custom_cam, pretrained_cam)
        # return self._overlay_heatmap(combined_heatmap, img_array[0])

        return self._overlay_heatmap(custom_cam, img_array[0])

    def _grad_cam(self, img_array, layer_name):
      try:
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(layer_name).output, self.model.output],
        )
      except ValueError:
        raise ValueError(f"Layer {layer_name} not found in model. Available layers: {[l.name for l in self.model.layers]}")

      with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

      grads = tape.gradient(loss, conv_output)
      pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

      conv_output = conv_output[0]
      if len(conv_output.shape) == 3:
        conv_output = tf.expand_dims(conv_output, axis=0)
      
      pooled_grads = pooled_grads[..., tf.newaxis, tf.newaxis, :]

      heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
      heatmap = tf.squeeze(heatmap)
      heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

      return heatmap.numpy()

    def _overlay_heatmap(self, heatmap, img):
      img = (img.numpy() * 255).astype(np.uint8)
      if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
      
      heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
      heatmap = (heatmap * 255).astype(np.uint8)
      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

      overlayed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

      return overlayed_img

