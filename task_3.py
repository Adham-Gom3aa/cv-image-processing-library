import numpy as np
import cv2


# ---------- Load ----------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Convert RGB image to grayscale using weighted average
def to_grayscale(image):
    height, width, _ = image.shape
    gray = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray[i, j] = 0.299*r + 0.587*g + 0.114*b

    return gray.astype(np.uint8)


# ---------- Analysis ----------
def analyze_image(gray):
    mean = np.sum(gray) / (gray.shape[0] * gray.shape[1]) # Calculate average brightness of the image using mean 

    variance = np.sum((gray - mean) ** 2) / (gray.shape[0] * gray.shape[1]) # Measure image contrast using standard deviation
    std = np.sqrt(variance)

    return mean, std


# Classify image based on brightness and contrast statistics
def classify_image(mean, std):
    if mean < 80:
        return "dark"
    elif mean > 180:
        return "bright"
    elif std < 40:
        return "low_contrast"
    else:
        return "normal"


# Enhance dark image by redistributing intensity values
def histogram_equalization(gray):
    hist = np.zeros(256)

    for value in gray.flatten():
        hist[value] += 1

    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]

    output = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            output[i, j] = int(cdf_normalized[gray[i, j]] * 255)

    return output


# Adjust image brightness using nonlinear gamma transformation if gamma>1( broght image)  if gamma<1 (darken image)
def gamma_correction(gray, gamma=0.5):
    output = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            normalized = gray[i, j] / 255.0
            corrected = normalized ** gamma
            output[i, j] = int(corrected * 255)

    return output


# Stretch intensity range to improve image contrast
def contrast_stretch(gray):
    min_val = np.min(gray)
    max_val = np.max(gray)

    output = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            output[i, j] = int(
                (gray[i, j] - min_val) / (max_val - min_val + 1e-5) * 255
            )

    return output


# ---------- Main Pipeline ----------
def auto_enhance(image):
    gray = to_grayscale(image)

    mean, std = analyze_image(gray)
    category = classify_image(mean, std)

    if category == "dark":
        enhanced = histogram_equalization(gray)
    elif category == "bright":
        enhanced = gamma_correction(gray, gamma=0.5)
    elif category == "low_contrast":
        enhanced = contrast_stretch(gray)
    else:
        enhanced = gray

    return enhanced, category
