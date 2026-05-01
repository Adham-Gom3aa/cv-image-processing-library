# cv-image-processing-library

A modular computer vision library built from scratch using NumPy and OpenCV for a series of practical image processing tasks.

---

## Task 1 — Selective Object Enhancement (Color-Based Editing Tool)

### Overview
A system that enhances a specific colored object in an image (sharpening) while applying a blurring effect to the rest of the image.

### Pipeline
```
Input → HSV Conversion → Color Mask → Noise Removal → Sharpen (mask) + Blur (background) → Output
```

### How it works

**1. Color Mask (`color_mask`)**
- Converts the image from RGB to HSV color space (better color separation than RGB)
- Creates a binary mask for the target color using predefined HSV bounds
- Red requires two HSV ranges (0–10 and 170–180) to cover its full hue wrap-around
- Removes small noise regions by filtering contours below `min_area`

**2. Blur (`blur_image`)**
- Box blur using a normalized k×k kernel
- Each pixel becomes the average of its k×k neighborhood
- Implemented with `sliding_window_view` for vectorized computation (no loops)

**3. Sharpen (`sharpen_image`)**
- Sobel edge detection using Gx and Gy kernels to detect horizontal and vertical edges
- Sharpened = Original + strength × (Gx + Gy)
- `strength` parameter controls how aggressive the sharpening is

**4. Selective Enhancement (`selective_enhancement`)**
- Pixels inside the mask → sharpened version
- Pixels outside the mask → blurred version
- Combined into a single output image

### Usage
```python
import task_1 as t1

image = t1.load_image("image.png")

# Sharpen red objects, blur everything else
result = t1.selective_enhancement(image, color='red', blur_kernel=20, min_area=500, strength=0.5)

# Supported colors: red, orange, yellow, green, cyan, blue, purple, pink
```

### Supported Colors
`red` · `orange` · `yellow` · `green` · `cyan` · `blue` · `purple` · `pink`

### Results
| Color Target | Effect |
|---|---|
| Red (roses) | Roses sharpened, background blurred |
| Yellow (sunflowers) | Sunflowers sharpened, background blurred |
| Green (stems) | Stems sharpened, background blurred |

---

## Task 2 — Comparative Study of Sharpening Pipelines for X-Ray Images

### Overview
Three independent image enhancement pipelines for X-ray images, each using a different strategy. Results are compared to analyze the strengths and weaknesses of each approach.

### Pipeline Overview
```
Input X-ray → Pipeline 1 / Pipeline 2 / Pipeline 3 → Side-by-side Comparison
```

### Pipeline 1 — Unsharp Masking

**Concept:**
Blur the image with a Gaussian filter → subtract the blurred from the original to isolate edges → add the edge mask back to amplify sharpness.

**Formula:**
```
mask   = original - gaussian_blurred
output = original + strength × mask
```

**Steps:**
1. Build a normalized 2D Gaussian kernel
2. Convolve image with Gaussian kernel to get blurred version
3. Subtract blurred from original → unsharp mask (edges + details only)
4. Add mask back scaled by `strength`
5. Clip and convert to uint8


### Pipeline 2 — CLAHE + Laplacian Sharpening

**Concept:**
Fix contrast region by region using CLAHE, then sharpen edges using a Laplacian kernel.

**Stage A — CLAHE (Contrast Limited Adaptive Histogram Equalization):**
1. Divide image into tiles (default 8×8 = 64 tiles)
2. For each tile: build histogram → clip at `clip_limit` → redistribute excess → build CDF → create LUT
3. Apply LUTs using bilinear interpolation between tile centers for smooth transitions

**Stage B — Laplacian Sharpening:**
```
output = CLAHE_result + strength × Laplacian(CLAHE_result)
```
Laplacian kernel:
```
[ 0, -1,  0 ]
[-1,  4, -1 ]
[ 0, -1,  0 ]
```

### Pipeline 3 — FFT Gaussian High-Pass Filter

**Concept:**
Convert image to frequency domain → suppress low frequencies (smooth regions) → boost high frequencies (edges and details) → convert back.

**Steps:**
1. Apply 2D FFT and shift DC component to center
2. Build Gaussian High-Pass Filter: `HPF = 1 - exp(-D² / (2 × cutoff²))`
3. Blend filter: `combined = 1 + boost × HPF` (keeps low freqs, amplifies high freqs)
4. Multiply FFT by filter → inverse FFT → take magnitude
5. Clip and convert to uint8

### Strengths & Weaknesses

| Pipeline | Strengths | Weaknesses |
|---|---|---|
| P1: Unsharp Masking | Simple, fast, easy to tune, preserves brightness | Cannot fix low contrast, not adaptive, amplifies noise at high strength |
| P2: CLAHE + Laplacian | Best for dark/low-contrast X-rays, adaptive per-tile, powerful combo | More parameters, tile artifacts at high clip_limit, Laplacian may amplify noise |
| P3: FFT High-Pass | Precise frequency control, reveals fine textures, principled approach | Ringing artifacts near edges, less intuitive parameters |

### Usage
```python
import task_2 as t2

xray = t2.load_image("chest_xray.png")

p1 = t2.pipeline_unsharp_masking(xray, blur_size=5, sigma=1.0, strength=1.5)
p2 = t2.pipeline_clahe_laplacian(xray, clip_limit=2.0, tile_grid_size=(8, 8), strength=1.0)
p3 = t2.pipeline_fft_highpass(xray, cutoff=30, boost=2.0)
```

### Output
![Pipeline Comparison](output.png)

---

## Requirements

```bash
pip install numpy opencv-python matplotlib
```

## File Structure

```
cv-image-processing-library/
├── task_1.py         # Task 1 — Selective Object Enhancement
├── task_2.py         # Task 2 — X-Ray Sharpening Pipelines
├── tests.ipynb       # Testing notebook for all tasks
└── README.md
```
