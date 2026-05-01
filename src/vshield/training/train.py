import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from vshield.models.cnn import CNNModel
from vshield.data.loader import load_data_from_config

def main():
    model_wrapper = CNNModel()
    
    # Đọc data từ file config yaml
    data = load_data_from_config("configs/data.yaml")
    
    if data["X_train"].size > 0:
        print("Bắt đầu training mô hình Anti-Spoofing...")
        model_wrapper.train(data=data, epochs=25, batch_size=16)
        
        # Lưu model
        model_wrapper.save("artifacts/models/face_verify_v1.keras")
        
        # Đánh giá trên tập val
        if data["X_val"].size > 0:
            print("Đánh giá mô hình trên tập validation:")
            model_wrapper.evaluate(data["X_val"], data["y_val"])
    else:
        print("Không tìm thấy dữ liệu training!")

if __name__ == "__main__":
    main()
