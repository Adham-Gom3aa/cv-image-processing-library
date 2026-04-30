import numpy as np
import cv2
import os

def load_image(path):
    """Load an image and convert to grayscale (X-ray images are grayscale (no color))."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image from: {path}")
    return img


def apply_kernel(image, kernel):
    """
    Manually convolve a 2D grayscale image with a given kernel using stride tricks.
    """
    h, w = image.shape # original image dimensions
    kh, kw = kernel.shape # kernel dimensions
    pad_h, pad_w = kh // 2, kw // 2 # padding size is half the kernel size 

    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect') # apply padding , reflect:mirrors the pixels at the border
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw)) # Creates a view where every pixel position has its own kh×kw neighborhood extracted 
    result = np.sum(windows * kernel, axis=(2, 3)) # Multiplies each neighborhood element wise with the kernel, then sums the result
    return result[:h, :w] 


def clip_and_convert(image):
    """just ensures no pixel goes below 0 or above 255 then covert from float to uint8 (int)"""
    return np.clip(image, 0, 255).astype(np.uint8)


#  Pipeline 1 : Unsharp Masking

def _gaussian_kernel(size, sigma):
    """Build a normalized 2D Gaussian kernel."""
    ax = np.arange(size) - size // 2 # creates a 1D array of coordinates centered at zero
    xx, yy = np.meshgrid(ax, ax)  # Expands that into 2D grids , xx holds the horizontal distances, yy holds the vertical distances from the center.
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) # Applies the Gaussian formula at every position
    return (kernel / kernel.sum()).astype(np.float32)


def pipeline_unsharp_masking(image, blur_size=5, sigma=1.0, strength=1.5):
  
    img_float = image.astype(np.float32) # convert to float for precision during calculations
    kernel = _gaussian_kernel(blur_size, sigma) # creates a Gaussian kernel of the specified size 
    blurred = apply_kernel(img_float, kernel) # blurs the image by convolving it with the Gaussian kernel
    unsharp_mask = img_float - blurred # calculates the unsharp mask by subtracting the blurred image from the original to show the edges and details 
    sharpened = img_float + strength * unsharp_mask # Adds those edges back to the original image
    return clip_and_convert(sharpened) # Clips and converts back to uint8 for the final result.


# Pipeline 2 : CLAHE + Laplacian Sharpening
# Stage A (CLAHE) : fix the contrast
# Stage B (Laplacian) : sharpen the edges

def _apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # X-ray images are often low contrast, so we apply CLAHE to enhance local contrast before sharpening.
    # CLAHE is the same like histogram equalization but instead of doing it for the whole image, divide it into small tiles and do it separately for each tile.
    h, w = image.shape
    # Each tile will be responsible for enhancing a specific region of the image, and we will blend the results smoothly to avoid seams.
    tile_h = h // tile_grid_size[1]
    tile_w = w // tile_grid_size[0]

    # Pad image so it divides evenly into tiles
    pad_h = tile_grid_size[1] * tile_h - h # how much padding needed to make height divisible by tile height
    pad_w = tile_grid_size[0] * tile_w - w # how much padding needed to make width divisible by tile width
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect') # Pad the image using reflection
    ph, pw = padded.shape # new dimensions after padding

    # Build per-tile lookup tables (LUTs):It's simply a translation table for pixel values (OLD intensity→NEW intensity )
    rows, cols = tile_grid_size[1], tile_grid_size[0]
    luts = np.zeros((rows, cols, 256), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            # Extract each tile and compute its histogram
            tile = padded[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256)) # counts how many pixels have each intensity value from 0 to 255

            # Clip histogram and redistribute excess uniformly
            clip_count = int(clip_limit * tile.size / 256) # maximum count allowed per bin based on clip limit
            excess = np.sum(np.maximum(hist - clip_count, 0)) # total excess pixels that need to be redistributed
            hist = np.minimum(hist, clip_count) # clip the histogram so no bin exceeds the clip count
            hist += excess // 256

            # Build CDF (Cumulative Distribution Function) to LUT 
            cdf = np.cumsum(hist).astype(np.float32) # cumulative sum of the histogram gives us the CDF (each value = sum of all previous values)
            cdf = (cdf - cdf.min()) / (tile.size - cdf.min() + 1e-6) * 255.0 # Normalize CDF to [0, 255] to become the LUT
            luts[r, c] = cdf

    # Apply LUTs with bilinear interpolation between tile centers
    output = np.zeros((ph, pw), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            # Finds the center point of the current tile in pixel coordinates
            cy = int((r + 0.5) * tile_h)
            cx = int((c + 0.5) * tile_w)

            # Gets the center of the next tile (to the right and below)
            r2 = min(r + 1, rows - 1)
            c2 = min(c + 1, cols - 1)
            cy2 = int((r2 + 0.5) * tile_h)
            cx2 = int((c2 + 0.5) * tile_w)

            # Defines the boundaries of the current tile in pixel coordinates
            y0 = r * tile_h
            y1 = (r + 1) * tile_h
            x0 = c * tile_w
            x1 = (c + 1) * tile_w

            for y in range(y0, y1):
                for x in range(x0, x1):
                    # Calculates how far the pixel is from the current tile center toward the next tile center
                    wy = (y - cy) / (cy2 - cy + 1e-6) if cy2 != cy else 0.0
                    wx = (x - cx) / (cx2 - cx + 1e-6) if cx2 != cx else 0.0
                    wy = np.clip(wy, 0, 1)
                    wx = np.clip(wx, 0, 1)
                    #Gets what each of the 4 surrounding tiles would map this pixel's intensity to.
                    pix = padded[y, x]
                    v00 = luts[r,  c ][pix]
                    v01 = luts[r,  c2][pix]
                    v10 = luts[r2, c ][pix]
                    v11 = luts[r2, c2][pix]
                    #Combines the 4 LUT values based on how close the pixel is to each tile center
                    output[y, x] = (
                        (1 - wy) * (1 - wx) * v00 +
                        (1 - wy) *      wx  * v01 +
                             wy  * (1 - wx) * v10 +
                             wy  *      wx  * v11
                    )

    return np.clip(output[:h, :w], 0, 255).astype(np.uint8)


def pipeline_clahe_laplacian(image, clip_limit=2.0, tile_grid_size=(8, 8), strength=1.0):
    # Stage A: adaptive contrast enhancement
    contrast_enhanced = _apply_clahe(image, clip_limit, tile_grid_size)

    # Stage B: Laplacian sharpening on the contrast-enhanced result 
    # The Laplacian kernel compares each pixel to its 4 neighbors 
    # If a pixel is very different from its neighbors , result is high : edge detected
    laplacian_kernel = np.array(
        [[0, -1,  0],
         [-1, 4, -1],
         [0, -1,  0]], dtype=np.float32
    )
    img_float = contrast_enhanced.astype(np.float32)
    laplacian = apply_kernel(img_float, laplacian_kernel)
    sharpened = img_float + strength * laplacian # Adds the detected edges back to the contrast-enhanced image to further enhance details
    return clip_and_convert(sharpened)


#  Pipeline 3 : Frequency Domain Enhancement (FFT-based High-Pass Filter)

def pipeline_fft_highpass(image, cutoff=30, boost=2.0):
    
    img_float = image.astype(np.float32)
    h, w = img_float.shape

    # Step 1: converts the image from pixels into frequencies using FFT then moves the low frequencies to the center of the result
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)

    # Step 2: Build Gaussian High-Pass Filter
    cy, cx = h // 2, w // 2 # center coordinates
    u = np.arange(w) - cx # horizontal distances from center
    v = np.arange(h) - cy # vertical distances from center
    U, V = np.meshgrid(u, v) # creates 2D grids of those distances
    D2 = U ** 2 + V ** 2                                    # squared distance from center
    hpf = 1.0 - np.exp(-D2 / (2.0 * cutoff ** 2))       # Gaussian high-pass filter formula: values near the center (low frequencies) are close to 0, values far from center (high frequencies) approach 1

    # Step 3: Instead of completely removing low frequencies (which would destroy the image structure),
    # we keep them but amplify the high frequencies on top
    combined_filter = 1.0 + boost * hpf

    # Step 4: Apply filter and inverse FFT
    filtered_fft = fft_shifted * combined_filter # Multiplies the frequency representation of the image by the combined filter
    ifft_shifted = np.fft.ifftshift(filtered_fft) # Shift back the zero frequency to the original position
    result = np.fft.ifft2(ifft_shifted) # converts back to spatial domain
    result = np.abs(result)                 

    return clip_and_convert(result)

