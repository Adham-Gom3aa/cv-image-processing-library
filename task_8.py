import cv2
import numpy as np
import random

# ============================================================================
# STEP 1: IMAGE ALIGNMENT
# Aligns multiple images by finding translation shifts between them
# ============================================================================
def align_images(img_list):
    """Align all images to the first image using phase correlation"""
    aligned = [img_list[0]]
    h, w = img_list[0].shape[:2]
    ref_eq = cv2.equalizeHist(cv2.cvtColor(img_list[0], cv2.COLOR_BGR2GRAY))

    for i in range(1, len(img_list)):
        gray_eq = cv2.equalizeHist(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY))
        shift, _ = cv2.phaseCorrelate(gray_eq.astype(np.float32), ref_eq.astype(np.float32))
        shift_x, shift_y = int(round(shift[0])), int(round(shift[1]))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped = cv2.warpAffine(img_list[i], M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        aligned.append(warped)
        ref_eq = cv2.equalizeHist(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    
    # Crop 2 pixels from edges to remove border artifacts
    return [img[2:-2, 2:-2] for img in aligned]


# ============================================================================
# STEP 2: BUILD WEIGHT FUNCTION
# Creates weights that give more importance to mid-range pixel values
# ============================================================================
def build_weight_function():
    """Hat-shaped weight function - higher weight for mid-range pixel values"""
    return np.array([z if z <= 127 else 255 - z for z in range(256)], dtype=np.float32)


# ============================================================================
# STEP 3: SAMPLE PIXEL POSITIONS
# Selects random pixel positions for CRF estimation
# ============================================================================
def sample_pixel_positions(image_shape, num_samples=100):
    """Generate random pixel coordinates for sampling"""
    rows, cols = image_shape[:2]
    return [(random.randint(0, rows-1), random.randint(0, cols-1)) for _ in range(num_samples)]


# ============================================================================
# STEP 4: RECOVER CAMERA RESPONSE FUNCTION (CRF)
# Uses Debevec's method to find mapping from pixel values to radiance
# ============================================================================
def recover_camera_response_function(images, exposures, smoothing_lambda=30):
    """
    Recover the Camera Response Function (CRF) g(Z) = ln(E) + ln(t)
    where Z is pixel value, E is radiance, t is exposure time
    """
    # Set random seed for reproducible results
    np.random.seed(41)
    random.seed(41)
    
    # Build weight function
    weights = build_weight_function()
    
    # Sample pixel positions
    rows, cols = images[0].shape[:2]
    indices = [(random.randint(0, rows-1), random.randint(0, cols-1)) for _ in range(100)]
    
    num_exposures = len(images)
    num_samples = 100
    
    # Build the linear system A * x = b
    A = np.zeros((num_samples * num_exposures + 255, 256 + num_samples), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)

    # Part 1: Data fitting term (pixel values to radiance relationship)
    k = 0
    for i in range(num_samples):
        for j in range(num_exposures):
            z = images[j][indices[i]]
            wij = weights[z]
            A[k, z] = wij
            A[k, 256 + i] = -wij
            b[k] = wij * np.log(exposures[j])
            k += 1

    # Part 2: Fix the curve at the midpoint (ensure stability)
    A[k, 127] = 1
    k += 1

    # Part 3: Smoothing term (encourages smooth CRF)
    for i in range(1, 255):
        A[k, i-1] = smoothing_lambda * weights[i]
        A[k, i] = -2 * smoothing_lambda * weights[i]
        A[k, i+1] = smoothing_lambda * weights[i]
        k += 1

    # Solve the least squares system
    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    return solution[:256].reshape(-1)


# ============================================================================
# STEP 5: MERGE EXPOSURES INTO HDR RADIANCE MAP
# Combines multiple exposures using the recovered CRF
# ============================================================================
def merge_to_hdr(images, exposures, crf_curves):
    """
    Merge multiple LDR images into a single HDR radiance map
    Uses weighted combination based on pixel reliability
    """
    h, w, c = images[0].shape
    hdr_radiance = np.zeros((h, w, c), dtype=np.float32)
    weights = build_weight_function()

    # Process each color channel independently
    for channel in range(c):
        crf = crf_curves[channel]
        sum_radiance = np.zeros((h, w), dtype=np.float32)
        sum_weights = np.zeros((h, w), dtype=np.float32)

        # Accumulate weighted radiance from all exposures
        for i in range(len(images)):
            img_channel = images[i][:, :, channel]
            # Convert pixel value to radiance: E = exp(g(Z) - ln(t))
            radiance = np.exp(crf[img_channel] - np.log(exposures[i]))
            weight_map = weights[img_channel]
            sum_radiance += weight_map * radiance
            sum_weights += weight_map

        # Average radiance (avoid division by zero)
        hdr_radiance[:, :, channel] = sum_radiance / (sum_weights + 1e-6)

    return hdr_radiance


# ============================================================================
# STEP 6: TONE MAPPING (REINHARD METHOD)
# Converts HDR radiance to displayable LDR image
# ============================================================================
def apply_reinhard_tone_mapping(hdr_radiance):
    """
    Apply Reinhard's global tone mapping operator
    Maps HDR radiance to [0,1] range for display
    """
    eps = 1e-6
    
    # Remove invalid values
    hdr_radiance = np.nan_to_num(hdr_radiance, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Compute luminance using standard coefficients
    luminance = (0.2126 * hdr_radiance[:,:,2] + 
                 0.7152 * hdr_radiance[:,:,1] + 
                 0.0722 * hdr_radiance[:,:,0])
    
    # Compute log-average luminance (key value)
    log_average_luminance = np.exp(np.mean(np.log(np.maximum(luminance, eps))))
    
    # Scale brightness based on key value (0.09 = moderately bright)
    key_value = 0.09
    scaled_radiance = (key_value / max(log_average_luminance, eps)) * hdr_radiance
    
    # Reinhard tone mapping formula: L_out = L_in / (1 + L_in)
    tone_mapped = scaled_radiance / (1.0 + scaled_radiance)
    
    # Compute luminance of tone-mapped image
    tone_mapped_luminance = (0.2126 * tone_mapped[:,:,2] + 
                             0.7152 * tone_mapped[:,:,1] + 
                             0.0722 * tone_mapped[:,:,0])
    
    # Apply saturation boost (bring colors back after compression)
    saturation = 1.0
    tone_mapped_luminance_3d = tone_mapped_luminance[:,:,np.newaxis]
    tone_mapped = np.clip(tone_mapped_luminance_3d + saturation * (tone_mapped - tone_mapped_luminance_3d), 0, 1)
    
    # Apply gamma correction for display
    gamma = 1.8
    tone_mapped = np.power(np.clip(tone_mapped, eps, None), 1.0 / gamma)
    
    # Convert to 8-bit image
    return np.uint8(np.clip(tone_mapped * 255, 0, 255))


# ============================================================================
# STEP 7: COMPLETE HDR PIPELINE
# Puts all steps together to process a set of bracketed exposures
# ============================================================================
def run_hdr_pipeline(image_files, exposure_times):
    """
    Complete HDR pipeline:
    1. Load images
    2. Align images
    3. Recover CRF for each color channel
    4. Merge exposures to HDR
    5. Apply tone mapping
    """
    # Step 1: Load all images
    images = [cv2.imread(f) for f in image_files]
    
    # Step 2: Align images to remove camera movement
    aligned_images = align_images(images)
    
    # Step 3: Recover CRF for each color channel (Blue, Green, Red)
    crf_curves = []
    for channel in range(3):  # B, G, R
        channel_images = [cv2.split(img)[channel] for img in aligned_images]
        crf = recover_camera_response_function(channel_images, exposure_times)
        crf_curves.append(crf)
    
    # Step 4: Merge exposures into HDR radiance map
    hdr_radiance = merge_to_hdr(aligned_images, exposure_times, crf_curves)
    
    # Step 5: Apply tone mapping to produce displayable image
    final_image = apply_reinhard_tone_mapping(hdr_radiance)
    
    return final_image
