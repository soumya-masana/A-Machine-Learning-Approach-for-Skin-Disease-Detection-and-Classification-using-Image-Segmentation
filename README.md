This project implements a machine learning-based system for automatic detection and classification of skin diseases from dermoscopic images using advanced image processing and segmentation techniques.

**Key Features**

**Comprehensive Preprocessing Pipeline:**

Digital hair removal using morphological Black-Hat transformation and inpainting

Noise reduction with Gaussian filtering

Image standardization (resizing to 512×512 pixels)

**Advanced Segmentation:**

Automatic GrabCut algorithm for precise lesion isolation

k-means clustering for improved segmentation

**Feature Extraction:**

Texture analysis using Gray Level Co-occurrence Matrix (GLCM)

Statistical features (mean, variance, standard deviation, RMS)

Color space transformations (RGB, HSV, LAB)

**Machine Learning Classification:**

Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Decision Tree classifiers

Multiclass handling for 8 skin disease types

Random oversampling to address class imbalance

**Performance:**

Achieves up to 97% accuracy on benchmark datasets (ISIC 2019 and HAM10000)

Comprehensive evaluation metrics (precision, recall, F1-score, ROC-AUC)

**Technical Specifications**

Programming Language: Python

Libraries: OpenCV, scikit-learn, scikit-image, NumPy, Pandas

Frameworks: Flask for web deployment

Datasets: ISIC 2019 Challenge, HAM10000

**Applications**

This system can be deployed as:

Clinical decision support tool for dermatologists

Mobile health application for preliminary skin screening

Educational tool for medical students

**Repository Structure**

<img width="637" height="166" alt="Screenshot 2025-07-29 172154" src="https://github.com/user-attachments/assets/9548f043-be05-497f-a675-522bb4296889" />

**Getting Started**

Clone the repository

Install dependencies: pip install -r requirements.txt

Run the application: python app.py

Access the web interface at http://localhost:5000

**Future Enhancements**

Integration of deep learning models (CNNs, Transformers)

Improved segmentation with U-Net or Mask R-CNN

Mobile app deployment

Explainable AI features for clinical interpretability
