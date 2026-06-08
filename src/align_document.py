import cv2
import numpy as np

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: Trái-Trên, Phải-Trên, Phải-Dưới, Trái-Dưới"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_aligned_paper(image: np.ndarray, target_w: int = 800, target_h: int = 1200):
    """
    Quy trình "Bàn ủi kỹ thuật số" (Document Alignment)
    Đọc ảnh, tìm 4 góc giấy (hoặc dò 4 ô vuông đen dự phòng) và nắn phẳng về kích thước chuẩn.
    """
    orig_h, orig_w = image.shape[:2]
    total_area = orig_h * orig_w
    
    # 1. Làm mờ và tìm cạnh (Canny) để bắt viền giấy
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    rect = None
    
    # 2. Thử tìm viền ngoài của tờ giấy thi (Khung giấy)
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Khung tìm được phải có 4 cạnh và chiếm ít nhất 30% diện tích ảnh
            if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * total_area:
                rect = order_points(approx.reshape(4, 2))
                break
                
    # 3. FALLBACK: Nếu chụp mất viền giấy -> Dò tìm 4 ô vuông đen (Anchors) ở 4 góc
    if rect is None:
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        cnts_anchor, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_anchors = []
        
        for c in cnts_anchor:
            x, y, wb, hb = cv2.boundingRect(c)
            aspect_ratio = wb / float(hb) if hb != 0 else 0
            area = cv2.contourArea(c)
            
            # Nhận dạng ô vuông: Tỷ lệ dài/rộng ~1 và diện tích 150-2000 px
            if 0.8 <= aspect_ratio <= 1.2 and 150 < area < 2000:
                valid_anchors.append([x + wb//2, y + hb//2])
        
        # Nếu tìm thấy từ 4 ô vuông trở lên (vì form này có đến 8 ô)
        # Lọc ra 4 ô ở vị trí góc ngoài cùng
        if len(valid_anchors) >= 4:
            pts = np.array(valid_anchors, dtype="float32")
            rect = np.zeros((4, 2), dtype="float32")
            
            # Tổng (x + y)
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Trái-Trên (Tổng x+y nhỏ nhất)
            rect[2] = pts[np.argmax(s)]  # Phải-Dưới (Tổng x+y lớn nhất)
            
            # Hiệu (y - x)
            diff = np.diff(pts, axis=1)  
            rect[1] = pts[np.argmin(diff)]  # Phải-Trên (y-x nhỏ nhất -> x lớn, y nhỏ)
            rect[3] = pts[np.argmax(diff)]  # Trái-Dưới (y-x lớn nhất -> x nhỏ, y lớn)
        else:
            print(f"❌ Chỉ tìm thấy {len(valid_anchors)} ô neo, không đủ 4 góc!")

    # 4. Kéo giãn ảnh (Warp Perspective - "Ủi phẳng")
    if rect is not None:
        dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (target_w, target_h))
        
    # 5. Nếu thất bại cả 2 cách, đành resize chay
    return cv2.resize(image, (target_w, target_h))