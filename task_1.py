import numpy as np
import cv2

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def blur_image(image, kernel_size=3):
    height, width, channels = image.shape
    pad = kernel_size // 2
    padded_image = np.zeros((height + 2 * pad, width + 2 * pad, channels), dtype=np.float32)
    padded_image[pad:pad + height, pad:pad + width] = image.astype(np.float32)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    output_image = np.zeros((height, width, channels), dtype=np.float32)
    
    for c in range(channels):
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_image[:, :, c], (kernel_size, kernel_size)
        )
        result = np.sum(windows * kernel, axis=(2, 3))
        result = result[:height, :width]
        
        output_image[:, :, c] = result
    
    return np.clip(output_image, 0, 255).astype(np.uint8)


def sharpen_image(image, strength=0.5):
    kernel_size = 3
    height, width, channels = image.shape
    img_float = image.astype(np.float32)
    pad = kernel_size // 2
    
    padded_image = np.zeros((height + 2 * pad, width + 2 * pad, channels), dtype=np.float32)
    padded_image[pad:pad + height, pad:pad + width] = img_float
    
    gx_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gy_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    
    output_image = np.zeros_like(img_float)
    
    for c in range(channels):
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_image[:, :, c], (kernel_size, kernel_size)
        )

        gx = np.sum(windows * gx_kernel, axis=(2, 3))
        gy = np.sum(windows * gy_kernel, axis=(2, 3))
        
        gx = gx[:height, :width]
        gy = gy[:height, :width]

        output_image[:, :, c] = img_float[:, :, c] + strength * (gx + gy)
    
    return np.clip(output_image, 0, 255).astype(np.uint8)


def color_mask(image, color, min_area=500):
    bounds = {
        "red": ([0, 50, 50], [10, 255, 255]),
        "orange": ([10, 50, 50], [20, 255, 255]),
        "yellow": ([20, 50, 50], [30, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "cyan": ([80, 50, 50], [100, 255, 255]),
        "blue": ([100, 50, 50], [130, 255, 255]),
        "purple": ([130, 50, 50], [160, 255, 255]),
        "pink": ([160, 50, 50], [180, 255, 255])
    }
    
    if color not in bounds:
        raise ValueError(f"Color '{color}' not found. Options: {list(bounds.keys())}")
    
    lower_bound, upper_bound = bounds[color]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
    
    if color == "red":
        mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    
    return mask


def selective_enhancement(image,color , min_area=500, strength=0.5, blur_kernel=3):
    mask = color_mask(image,color, min_area)
    blurred_image = blur_image(image, blur_kernel)
    sharpened_image = sharpen_image(image, strength)
    output_image = image.copy()
    output_image[mask > 0] = sharpened_image[mask > 0]
    output_image[mask == 0] = blurred_image[mask == 0]
    return output_image