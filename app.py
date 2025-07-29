import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from utils import preprocess_image, extract_features

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained models
MODEL_PATHS = {
    'svm': 'models/svm_classifier.pkl',
    'knn': 'models/knn_classifier.pkl',
    'dt': 'models/decision_tree_classifier.pkl'
}

CLASS_NAMES = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and extract features
            preprocessed_img = preprocess_image(filepath)
            features = extract_features(preprocessed_img)

            # Scale features
            scaler = joblib.load('models/scaler.pkl')
            features_scaled = scaler.transform([features])

            # Before prediction, validate features
            features = extract_features(preprocessed_img)
            print("Features:", features)  # Should differ per image

            if np.all(features == features[0]):  # All values identical
                raise ValueError("Feature extraction failed!")

            # Make predictions with all classifiers
            predictions = {}
            for model_name, model_path in MODEL_PATHS.items():
                model = joblib.load(model_path)
                pred = model.predict(features_scaled)[0]
                predictions[model_name] = {
                    'class': CLASS_NAMES[pred],
                    'confidence': np.max(model.predict_proba(features_scaled)) * 100
                }

            # Get segmented image path
            segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_' + filename)
            cv2.imwrite(segmented_path, preprocessed_img)

            return render_template('index.html',
                                   original_img=filepath,
                                   segmented_img=segmented_path,
                                   predictions=predictions)

    return render_template('index.html')


def extract_features(image):
    """Example with 30 fixed features"""
    # 1. GLCM Features (6)
    glcm_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Replace with actual calculations

    # 2. Color Statistics (24)
    color_stats = [0.01] * 24  # Replace with actual calculations

    # Combine to make 30 features total
    return np.array(glcm_features + color_stats)

if __name__ == '__main__':
    app.run(debug=True)


