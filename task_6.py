import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(reference_path, query_path):
    """Load images and convert BGR to RGB."""
    ref = cv2.cvtColor(cv2.imread(reference_path), cv2.COLOR_BGR2RGB)
    query = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    return ref, query

def crop_object(img):
    """Auto-crops an image tightly around its edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    
    # findNonZero grabs all edge pixels, boundingRect wraps them in one box
    pts = cv2.findNonZero(dilated)
    if pts is None: 
        return img
    x, y, w, h = cv2.boundingRect(pts)
    return img[y:y+h, x:x+w]

def extract_sift_features(ref, query):
    """Convert to grayscale and extract SIFT keypoints and descriptors."""
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), None)
    kp2, desc2 = sift.detectAndCompute(cv2.cvtColor(query, cv2.COLOR_RGB2GRAY), None)
    return kp1, desc1, kp2, desc2

def match_features(desc1, desc2, ratio=0.75):
    """Match descriptors using FLANN and Lowe's ratio test."""
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Filter for good matches
    return [m for m, n in matches if m.distance < ratio * n.distance]

def compute_homography(kp1, kp2, good_matches):
    """Calculate the perspective transformation matrix (M)."""
    if len(good_matches) < 4:
        raise ValueError(f"Only {len(good_matches)} matches found (need 4+).")
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

def transform_and_draw(ref, query, M, good_matches):
    """Apply transformation to the bounding box and display the results."""
    h, w = ref.shape[:2]
    
    # Transform bounding box corners
    obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(obj_corners, M)
    
    # Draw boxes
    result = cv2.polylines(query.copy(), [transformed.astype(np.int32)], True, (0, 255, 0), 3)
    ref_with_box = cv2.rectangle(ref.copy(), (0, 0), (w, h), (0, 255, 0), 4)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(ref_with_box)
    ax1.set_title("Auto-Cropped Reference")
    ax1.axis('off')
    
    ax2.imshow(result)
    ax2.set_title(f"Detected Object ({len(good_matches)} matches)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

def recognize_object(reference_path, query_path):
    """Main orchestrator function for the pipeline."""
    # Step 1: Load
    ref, query = load_images(reference_path, query_path)
    
    # Step 2: Auto-Crop
    ref = crop_object(ref)
    
    # Step 3 & 4: Detect and Match
    kp1, desc1, kp2, desc2 = extract_sift_features(ref, query)
    good_matches = match_features(desc1, desc2)
    
    # Step 5: Compute math
    M = compute_homography(kp1, kp2, good_matches)
    
    # Step 6: Draw
    result = transform_and_draw(ref, query, M, good_matches)
    
    return result, len(good_matches)