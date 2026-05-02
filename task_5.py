import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1, img2, img1_rgb, img2_rgb, gray1, gray2


# SIFT
def detect_features(gray1, gray2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    return kp1, kp2, des1, des2


def match_features(des1, des2, ratio=0.7):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def compute_homography(kp1, kp2, good_matches):
    pts1 = []
    pts2 = []

    for m in good_matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    return H, mask


def stitch_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    corners_img1 = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(corners_img2, H)

    all_corners = np.concatenate((corners_img1, transformed), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-x_min, -y_min]

    H_translation = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    result = cv2.warpPerspective(
        img2,
        H_translation.dot(H),
        (x_max - x_min, y_max - y_min)
    )

    result[
        translation[1]:h1 + translation[1],
        translation[0]:w1 + translation[0]
    ] = img1

    return result


# Full Pipeline
def panorama_pipeline(path1, path2):
    img1, img2, img1_rgb, img2_rgb, gray1, gray2 = load_images(path1, path2)

    kp1, kp2, des1, des2 = detect_features(gray1, gray2)

    good_matches = match_features(des1, des2)

    H, mask = compute_homography(kp1, kp2, good_matches)

    result = stitch_images(img1, img2, H)

    return img1_rgb, img2_rgb, result
