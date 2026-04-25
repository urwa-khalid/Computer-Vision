# algorithms.py
import cv2
import numpy as np
import math

def to_gray(image):
    """Converts image to grayscale for processing."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def rotation(image, angle_deg):
    """Assignment 2: Custom 360 Degree Rotation Logic"""
    height, width = image.shape[:2]
    result = np.zeros_like(image)
    
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx, cy = width // 2, height // 2 

    for y in range(height):
        for x in range(width):
            tx, ty = x - cx, y - cy
            # Rotation Matrix Calculation
            nx = int(tx * cos_a - ty * sin_a + cx)
            ny = int(tx * sin_a + ty * cos_a + cy)
            # Boundary check
            if 0 <= nx < width and 0 <= ny < height:
                result[ny, nx] = image[y, x]
    return result

def gray_quantization(image, levels, mode):
    """Assignment 3: Gray-level Quantization"""
    gray = to_gray(image).astype(np.int32)
    step = 256 // levels
    result = np.zeros_like(gray, dtype=np.uint8)
    for i in range(0, 256, step):
        mask = (gray >= i) & (gray < i + step)
        if mode == "Lower":
            result[mask] = np.uint8(i)
        elif mode == "Higher":
            result[mask] = np.uint8(min(i + step - 1, 255))
        elif mode == "Middle":
            result[mask] = np.uint8(i + step // 2)
    return result

def igs_quantization(image, levels):
    """Assignment 3: Improved Gray Scale (IGS) Quantization"""
    gray = to_gray(image).astype(np.int32)
    rows, cols = gray.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    prev_error = 0
    bits_to_keep = int(math.log2(levels))
    shift = 8 - bits_to_keep

    for i in range(rows):
        for j in range(cols):
            val = gray[i, j] + prev_error
            if val > 255: val = 255
            quantized = (val >> shift) << shift
            result[i, j] = quantized
            prev_error = val & ((1 << shift) - 1)
    return result