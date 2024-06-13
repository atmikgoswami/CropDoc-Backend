import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

model = pickle.load(open('modeld.pkl', 'rb'))

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

diseases = [
'apple apple scab',
'apple black rot',
'apple cedar apple rust',
'apple healthy',
'blueberry healthy',
'cherry including sour powdery mildew',
'cherry including sour healthy',
'corn maize cercospora leaf spot gray leaf spot',
'corn maize common rust ',
'corn maize northern leaf blight',
'corn maize healthy',
'grape black rot',
'grape esca black measles' ,
'grape leaf blight isariopsis leaf spot ',
'grape healthy',
'orange haunglongbing citrus greening ',
'peach bacterial spot',
'peach healthy',
'pepper bell bacterial spot',
'pepper bell healthy',
'potato early blight',
'potato late blight',
'potato healthy',
'raspberry healthy',
'soybean healthy',
'squash powdery mildew',
'strawberry leaf scorch',
'strawberry healthy',
'tomato bacterial spot',
'tomato early blight',
'tomato late blight',
'tomato leaf mold',
'tomato septoria leaf spot',
'tomato spider mites two spotted spider mite',
'tomato target spot',
'tomato tomato yellow leaf curl virus',
'tomato tomato mosaic virus',
'tomato healthy'
]

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

        # Prepare input image
        input_image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        input_data = np.expand_dims(input_image, axis=0)
        input_data = input_data.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Obtain output results from the interpreter
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Example: Assuming output is probabilities for classification
        predicted_class_index = np.argmax(output_data)
        confidence = output_data[0][predicted_class_index]

        if confidence >= 0.4:
            predicted_disease = diseases[predicted_class_index]
        else:
            predicted_disease = "Healthy Leaf"

        return jsonify({'prediction': predicted_disease})

if __name__ == "__main__":
    app.run()
