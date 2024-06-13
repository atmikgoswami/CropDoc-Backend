from gevent import monkey
monkey.patch_all()

import torch
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

crop_mapping = {
    0: 'blackgram',
    1: 'chickpea',
    2: 'cotton',
    3: 'jute',
    4: 'kidneybeans',
    5: 'lentil',
    6: 'maize',
    7: 'mothbeans',
    8: 'muskmelon',
    9: 'mungbean',
    10: 'pigeonpeas',
    11: 'rice',
    12: 'watermelon',
}

diseases = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

class Plant_Disease_Model2(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(weights=None)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)
    
    def forward(self, xb):
        out = self.network(xb)
        return out

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()
plant_disease_model = Plant_Disease_Model2()
plant_disease_model.load_state_dict(torch.load('plantDisease-resnet34.pth', map_location=device))
plant_disease_model = to_device(plant_disease_model, device)
plant_disease_model.eval()

transform = transforms.Compose([transforms.Resize(size=128), transforms.ToTensor()])

@app.route('/')
def home():
    return 'Hello from Server'

@app.route('/recommend', methods=['POST'])
def predict():
    input_data = request.json
    temperature = input_data['temp']
    humidity = input_data['humidity']
    ph = input_data['ph']
    water = input_data['water']
    season = input_data['season']

    # Ensure the input data is in the correct format for the model
    input_features = np.array([[temperature, humidity, ph, water, season]], dtype=np.float32)

    # Make predictions
    prediction = model.predict(input_features)

    # Convert to crop name
    predicted_crop = crop_mapping[prediction[0]]

    return jsonify({'prediction': predicted_crop})

@app.route('/disease', methods=['POST'])
def upload_file():
    if 'crop' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['crop']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read image from request directly
        image = Image.open(file.stream)

        # Predict the disease
        transformed_image = transform(image)
        tensor_image = transformed_image.unsqueeze(0)
        img = tensor_image[0]
        predicted_disease = predict_image(img, plant_disease_model)
        return jsonify({'prediction': predicted_disease})

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return diseases[preds[0].item()]

if __name__ == "__main__":
    app.run()
