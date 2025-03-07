from inference.predictor import DiseasePredictor
from inference.xai import XAIProcess
import cv2

def full_diagnosis_pipeline(image_path, env_data):
    predictor = DiseasePredictor("best_model.keras")
    xai_processor = XAIProcess(predictor.model)

    prediction = predictor.predict(image_path)

    img = predictor.preprocess_image(image_path)
    explanation_img = xai_processor.multi_model_gradcam(img)

    prompt = predictor.generate_prompt(prediction, env_data)
    return explanation_img, prompt, prediction


if __name__ == "__main__":
    env_data = {
        "location": "maharashtra",
        "temp": 32,
        "humidity": '70%',
        "soil_status": "5.5pH, red soil",
        "weather": "moderate rainfall",
    }
    full_diagnosis_pipeline("../cotton_leaves/Aphids/1.jpg", env_data=env_data)
    img, prompt, pred = full_diagnosis_pipeline("../cotton_leaves/Aphids/1.jpg", env_data=env_data)
    
    cv2.imwrite("explanation.jpg", img)
    print("Generated prompt:", prompt)
    print("Prediction:", pred)