import json
import os

def generate_bubble_coordinates(output_json_path="data/template_config.json"):
    
    START_X = 141.0  
    START_Y = 343.0
    X_STRIDE = 30.5 
    Y_STRIDE = 23.7  
    COL_STRIDE = 179 
    BLOCK_GAP = 0    
    NUM_QUESTIONS = 40
    
    bubbles = []
    
    for q in range(NUM_QUESTIONS):
        col_index = q // 10  
        row_index = q % 10   
        
        center_y = START_Y + (row_index * Y_STRIDE)
        if row_index >= 5:
            center_y += BLOCK_GAP
            
        for ans_idx, ans_char in enumerate(['A', 'B', 'C', 'D']):
            # Tính tọa độ TÂM y hệt như lúc vẽ
            center_x = START_X + (ans_idx * X_STRIDE) + (col_index * COL_STRIDE)
            
            # Lưu tọa độ TÂM và Kích thước Box (32x32)
            bubbles.append({
                "question": q + 1,
                "option": ans_char,
                "x": int(center_x),
                "y": int(center_y),
                "w": 32,
                "h": 32
            })
            
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({"bubbles": bubbles}, f, indent=4)
        
    print(f"✅ Đã tạo file tọa độ gốc tại: {output_json_path}")

if __name__ == "__main__":
    generate_bubble_coordinates()