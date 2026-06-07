import cv2
import numpy as np

def detect_template_anchors(binary_image):
    """Tự động phát hiện 4 góc (anchors) của phiếu thi để áp dụng Perspective Transform"""
    
    inv = cv2.bitwise_not(binary_image)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    h, w = binary_image.shape
    
    quadrants = {
        'top_left': (0, 0, w//2, h//2),
        'top_right': (w//2, 0, w, h//2),
        'bottom_left': (0, h//2, w//2, h),
        'bottom_right': (w//2, h//2, w, h),
    }
    
    for quad_name, (qx1, qy1, qx2, qy2) in quadrants.items():
        for c in cnts:
            x, y, wb, hb = cv2.boundingRect(c)
            if qx1 <= x <= qx2 and qy1 <= y <= qy2:
                if 150 < cv2.contourArea(c) < 900:  # Diện tích đủ khớp với ô đen anchor
                    corners.append({'quad': quad_name, 'x': x + wb//2, 'y': y + hb//2})
                    break
    
    return corners  