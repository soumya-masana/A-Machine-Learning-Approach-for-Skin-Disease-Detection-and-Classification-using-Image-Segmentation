import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans


def remove_hairs(image):
    """Remove hairs using black-hat transform and inpainting"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black hat transform to find hairs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the blackhat image to get hair mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the original image using the mask
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return inpainted


def apply_gaussian_filter(image):
    """Apply Gaussian filter for noise reduction"""
    return cv2.GaussianBlur(image, (5, 5), 0)


def grabcut_segmentation(image):
    """Segment lesion using GrabCut algorithm"""
    mask = np.zeros(image.shape[:2], np.uint8)

    # Background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define ROI (adjust based on your needs)
    height, width = image.shape[:2]
    rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply mask to image
    segmented = image * mask2[:, :, np.newaxis]

    return segmented, mask2


def kmeans_clustering(image, k=3):
    """Apply K-means clustering for feature extraction"""
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Convert to float32 for K-means
    pixels = np.float32(pixels)

    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to centers
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def extract_glcm_features(image):
    """Extract GLCM texture features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)

    # Extract properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]

    return [contrast, dissimilarity, homogeneity, energy, correlation, asm]
    print("GLCM Features:", glcm_features)

def extract_color_features(image):
    """Extract statistical color features"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Calculate mean and std for each channel in each color space
    bgr_mean = np.mean(image, axis=(0, 1))
    bgr_std = np.std(image, axis=(0, 1))

    hsv_mean = np.mean(hsv, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))

    lab_mean = np.mean(lab, axis=(0, 1))
    lab_std = np.std(lab, axis=(0, 1))

    # Flatten all features into a single list
    color_features = np.concatenate([
        bgr_mean, bgr_std,
        hsv_mean, hsv_std,
        lab_mean, lab_std
    ])

    return color_features.tolist()
    print("Color Stats:", color_stats)

def preprocess_image(image_path):
    """Full preprocessing pipeline"""
    # Read image
    image = cv2.imread(image_path)

    # Hair removal
    image = remove_hairs(image)

    # Noise reduction
    image = apply_gaussian_filter(image)

    # Segmentation
    segmented, _ = grabcut_segmentation(image)

    # K-means clustering
    clustered = kmeans_clustering(segmented)

    return clustered


def extract_features(image):
    # Debug: Ensure image isn't blank
    if np.all(image == 0):
        raise ValueError("Blank image detected!")

    # Example GLCM calculation (replace with your actual code)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256)
    contrast = graycoprops(glcm, 'contrast')[0, 0]

    # Debug: Print dynamic values
    print(f"Contrast: {contrast:.2f}")  # Should vary per image

    return np.array([contrast, ...])  # Ensure 30 features total

    assert img is not None, "Image failed to load!"