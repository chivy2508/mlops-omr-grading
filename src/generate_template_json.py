import json
import os

def generate_bubble_coordinates(output_json_path="data/template_config.json"):
    
    START_X = 100.0    
    START_Y = 688.0     
    X_STRIDE = 30.0     
    Y_STRIDE = 52.4     
    COL_STRIDE = 180.0  
    BUBBLE_SIZE = 21    
    
    bubbles = []
    
    # 4 cột (1-10, 11-20, 21-30, 31-40)
    for col_index in range(4):
        # 10 câu mỗi cột
        for row_index in range(10):
            q_num = col_index * 10 + row_index + 1
            center_y = START_Y + (row_index * Y_STRIDE)
            
            # 4 đáp án A, B, C, D
            for ans_idx, ans_char in enumerate(['A', 'B', 'C', 'D']):
                center_x = START_X + (ans_idx * X_STRIDE) + (col_index * COL_STRIDE)
                
                bubbles.append({
                    "question": q_num,
                    "option": ans_char,
                    "x": int(center_x),
                    "y": int(center_y),
                    "w": BUBBLE_SIZE,
                    "h": BUBBLE_SIZE
                })
            
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({"bubbles": bubbles}, f, indent=4)
    print(f"✅ Đã tạo file template với {len(bubbles)} ô đáp án (Tọa độ mới)!")

if __name__ == "__main__":
    generate_bubble_coordinates()