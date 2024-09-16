import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize the Flask application
app = Flask(__name__)

# Set the folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the saved model
model_path = "mobile_net.h5"
model = tf.keras.models.load_model(model_path)

# Define the image size (ensure this matches the size used during training)
img_size = (240, 240)

# Assuming 'class_names' contains the names of the classes
class_names = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis',
    'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
    'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion'
]

def predict_disease(model, img_path):
    try:
        # Load and preprocess the image
        img = tf.keras.utils.load_img(img_path, target_size=img_size, color_mode='rgb')
        array = tf.keras.utils.img_to_array(img)
        array = array / 255.0

        img_array = np.expand_dims(array, axis=0)
        preds = model.predict(img_array)

        # Format predictions
        top_prob_index = np.argmax(preds[0])
        top_prob = round(preds[0][top_prob_index] * 100, 2)

        # Plot the image with the prediction
        plt.imshow(array)  # Using the preprocessed array instead of reloading the image
        plt.axis('off')
        plt.title(f"Class: {class_names[top_prob_index]}; Prob: {top_prob}%")
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()  # Close the current figure
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Return top prediction and image as base64 string
        return class_names[top_prob_index], top_prob, img_str
    except Exception as e:
        return None, None, str(e)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Dermatology ML Model API! Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            class_name, probability, img_str = predict_disease(model, file_path)

            if class_name is None:
                return jsonify({'error': 'Error during prediction', 'details': probability}), 500

            # Return the prediction in JSON format
            return jsonify({
                'predicted_class': class_name,
                'top_prob': probability,
                'image': img_str
            })
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
