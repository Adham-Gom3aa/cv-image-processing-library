import numpy as np
import cv2


def load_image(path):
    """Load an image and convert to grayscale (X-ray images are grayscale (no color))."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image from: {path}")
    return img


def clip_and_convert(image):
    """ensures no pixel goes below 0 or above 255 then converts from float to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)


# Pipeline 1 : Unsharp Masking
# 1. Blur the image using a Gaussian filter
# 2. Subtract the blurred from the original → isolates edges and details (unsharp mask)
# 3. Add the mask back scaled by strength → amplifies edges

def pipeline_unsharp_masking(image, blur_size=5, sigma=1.0, strength=1.5):

    img_float = image.astype(np.float32) # convert to float for precision during calculations
    blurred = cv2.GaussianBlur(img_float, (blur_size, blur_size), sigma) # blurs the image using a Gaussian kernel
    unsharp_mask = img_float - blurred # subtracts the blurred image from the original to isolate edges and details
    sharpened = img_float + strength * unsharp_mask # adds edges back to the original image
    return clip_and_convert(sharpened)


# Pipeline 2 : CLAHE + Laplacian Sharpening 
# Stage A (CLAHE)     : fix the contrast region by region
# Stage B (Laplacian) : sharpen the edges on top

def _apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # CLAHE is like histogram equalization but instead of doing it for the whole image,
    # it divides the image into small tiles and equalizes each tile independently.
    h, w = image.shape
    tile_h = h // tile_grid_size[1]
    tile_w = w // tile_grid_size[0]

    # Pad image so it divides evenly into tiles
    pad_h = tile_grid_size[1] * tile_h - h 
    pad_w = tile_grid_size[0] * tile_w - w 
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape

    # Build per-tile LUTs (Lookup Tables) : translation table : old intensity → new intensity
    rows, cols = tile_grid_size[1], tile_grid_size[0]
    luts = np.zeros((rows, cols, 256), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            tile = padded[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256)) # counts how many pixels have each intensity value from 0 to 255

            # Clip histogram and redistribute excess uniformly to prevent noise amplification
            clip_count = int(clip_limit * tile.size / 256) # maximum count allowed per bin
            excess = np.sum(np.maximum(hist - clip_count, 0)) # total excess pixels to redistribute
            hist = np.minimum(hist, clip_count) # clip any bin that exceeds the limit
            hist += excess // 256 # redistribute excess evenly across all bins

            # Build CDF → normalize to [0,255] → this becomes the LUT
            cdf = np.cumsum(hist).astype(np.float32) # cumulative sum (each value = sum of all previous)
            cdf = (cdf - cdf.min()) / (tile.size - cdf.min() + 1e-6) * 255.0
            luts[r, c] = cdf

    # Apply LUTs with bilinear interpolation between tile centers to avoid harsh borders
    output = np.zeros((ph, pw), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cy = int((r + 0.5) * tile_h) # center of current tile
            cx = int((c + 0.5) * tile_w)

            r2 = min(r + 1, rows - 1) # next tile (clamped to grid boundary)
            c2 = min(c + 1, cols - 1)
            cy2 = int((r2 + 0.5) * tile_h) # center of next tile
            cx2 = int((c2 + 0.5) * tile_w)

            y0, y1 = r * tile_h, (r + 1) * tile_h # pixel boundaries of current tile
            x0, x1 = c * tile_w, (c + 1) * tile_w

            for y in range(y0, y1):
                for x in range(x0, x1):
                    # How far this pixel is from current tile center toward next tile center
                    wy = (y - cy) / (cy2 - cy + 1e-6) if cy2 != cy else 0.0
                    wx = (x - cx) / (cx2 - cx + 1e-6) if cx2 != cx else 0.0
                    wy = np.clip(wy, 0, 1)
                    wx = np.clip(wx, 0, 1)

                    pix = padded[y, x]
                    v00 = luts[r,  c ][pix] # current tile LUT value
                    v01 = luts[r,  c2][pix] # right tile LUT value
                    v10 = luts[r2, c ][pix] # bottom tile LUT value
                    v11 = luts[r2, c2][pix] # bottom-right tile LUT value

                    # Blend 4 surrounding LUT values based on proximity to each tile center
                    output[y, x] = (
                        (1 - wy) * (1 - wx) * v00 +
                        (1 - wy) *      wx  * v01 +
                             wy  * (1 - wx) * v10 +
                             wy  *      wx  * v11
                    )

    return np.clip(output[:h, :w], 0, 255).astype(np.uint8)


def pipeline_clahe_laplacian(image, clip_limit=2.0, tile_grid_size=(8, 8), strength=1.0):
    # Stage A: fix contrast using CLAHE
    contrast_enhanced = _apply_clahe(image, clip_limit, tile_grid_size)

    # Stage B: Laplacian sharpening
    # compares each pixel to its 4 neighbors , high difference = edge detected
    laplacian = cv2.Laplacian(contrast_enhanced, cv2.CV_32F) # detects edges using the Laplacian kernel
    sharpened = contrast_enhanced.astype(np.float32) + strength * laplacian # add detected edges back to enhance details
    return clip_and_convert(sharpened)


# Pipeline 3 : FFT-based Gaussian High-Pass Filter 
# Low frequencies  = smooth regions, background, overall brightness
# High frequencies = edges, details, fine textures
# boost high frequencies to sharpen the image

def pipeline_fft_highpass(image, cutoff=30, boost=2.0):

    img_float = image.astype(np.float32)
    h, w = img_float.shape

    # convert image to frequency domain, shift DC (low freq) to center
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)

    # build Gaussian High-Pass Filter
    cy, cx = h // 2, w // 2 # center of frequency map
    u = np.arange(w) - cx # horizontal distances from center
    v = np.arange(h) - cy # vertical distances from center
    U, V = np.meshgrid(u, v)
    D2 = U ** 2 + V ** 2 # squared distance from center
    hpf = 1.0 - np.exp(-D2 / (2.0 * cutoff ** 2)) # near center (low freq) → ~0 , far from center (high freq) → ~1

    # keep low frequencies but amplify high frequencies
    combined_filter = 1.0 + boost * hpf

    # apply filter then convert back to spatial domain
    filtered_fft = fft_shifted * combined_filter
    ifft_shifted = np.fft.ifftshift(filtered_fft) # shift back zero frequency to original position
    result = np.fft.ifft2(ifft_shifted) # convert back to pixels
    result = np.abs(result) # take magnitude (FFT produces complex numbers)

    return clip_and_convert(result)
