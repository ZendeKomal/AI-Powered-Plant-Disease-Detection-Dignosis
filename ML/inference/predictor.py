import json 
import numpy as np 
import tensorflow as tf

class DiseasePredictor:
  def __init__(self, model_path):
    self.model = tf.keras.models.load_model(model_path,
      custom_objects={
        'Rescaling': tf.keras.layers.Rescaling,
        'Functional': tf.keras.models.Model
      }
    )

    with open('class_names.json', 'r') as f:
      self.class_names = json.load(f)

  def _calculate_severity(self, confidence):
    thresholds = [(0.9, 'Critical'), (0.7, 'Severe'), (0.5, 'Moderate'), (0.3, 'Mild')]
    for thresh, label in thresholds:
        if confidence > thresh:
          return label
    return 'Low'

  def preprocess_image(self, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, self.model.input_shape[1:3])

    return tf.expand_dims(img, axis=0)
  
  def predict(self, image_path):
    img = self.preprocess_image(image_path)
    probs = self.model.predict(img)
    class_id = np.argmax(probs)

    return {
      'plant_type': 'Cotton Plant',
      'disease': self.class_names[class_id],
      'confidence': float(probs[0][class_id]),
      'all_preobs': probs[0].tolist(),
      'severity': self._calculate_severity(probs[0][class_id])
    }
  
  def generate_prompt(self, prediction, environmental_data):
    return f"""
    As a cotton plant pathologist, provide comprehensive recommendations for:

    - Affected Crop: Cotton ({prediction['plant_type']})
    - Disease Identified: {prediction['disease']}
    - Infection Severity: {prediction['severity']}
    - Diagnostic Confidence: {prediction['confidence']*100:.1f}%
    - Environmental Context:
      * Location: {environmental_data['location']}
      * Temperature: {environmental_data['temp']}°C
      * Humidity: {environmental_data['humidity']}%
      * Soil Conditions: {environmental_data['soil_status']}
      * Recent Weather: {environmental_data['weather']}
    
    Provide recommendations addressing:
    1. Immediate containment protocols
    2. Approved fungicides/bactericides with application rates
    3. Cultural management practices
    4. Soil treatment recommendations
    5. Long-term crop rotation strategies
    6. Expected recovery progression
    7. Quarantine measures if needed
    """