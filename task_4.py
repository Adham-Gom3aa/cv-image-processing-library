import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def median_filter(image, k=21):
    h, w = image.shape
    pad = k // 2
    
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+k, j:j+k]
            output[i, j] = np.median(window)

    return output.astype(np.uint8)

def remove_background(gray, background):
    result = background.astype(np.int16) - gray.astype(np.int16)
    return np.clip(result, 0, 255).astype(np.uint8)

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    
    if max_val - min_val == 0:
        return img
    
    norm = (img - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

def invert(img):
    return 255 - img

def simple_threshold(image, t):
    h, w = image.shape
    out = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            if image[i, j] > t:
                out[i, j] = 255  
            else:
                out[i, j] = 0 
    return out

def document_pipeline(image):
    gray = to_grayscale(image)

    background = median_filter(gray, 21)

    cleaned = remove_background(gray, background)

    normalized = normalize(cleaned)

    inverted = invert(normalized)

    t = np.mean(inverted)

    binary = simple_threshold(inverted, t)

    return binary