import os
import cv2
import yaml
import numpy as np


def load_data_from_config(yaml_path):
    # 1. Đọc nội dung file yaml
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    project_root = os.path.dirname(os.path.abspath(yaml_path))
    base_path = os.path.join(project_root, config["path"])
    # Lấy các thông tin từ config
    base_path = config["path"]  # ./data/SplitData

    def get_data_from_split(split_key):
        images = []
        labels = []

        # Đường dẫn tới thư mục images: ./data/SplitData/train/images
        img_folder = os.path.join(base_path, config[split_key])
        # Tự động suy luận thư mục labels bằng cách thay 'images' thành 'labels'
        lbl_folder = img_folder.replace("images", "labels")

        print(f"Đang quét thư mục: {img_folder}")

        if not os.path.exists(img_folder):
            return np.array([]), np.array([])

        for img_name in os.listdir(img_folder):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                # Đọc và chuẩn hóa ảnh
                img_path = os.path.join(img_folder, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))
                images.append(img)

                # Tìm file label tương ứng
                lbl_name = os.path.splitext(img_name)[0] + ".txt"
                lbl_path = os.path.join(lbl_folder, lbl_name)

                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        # Lấy class_id từ dòng đầu tiên
                        class_id = int(f.readline().split()[0])
                        labels.append(class_id)
                else:
                    labels.append(0)  # Mặc định nếu thiếu label

        return np.array(images) / 255.0, np.array(labels)

    # Load dữ liệu cho từng bộ
    X_train, y_train = get_data_from_split("train")
    X_val, y_val = get_data_from_split("val")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "classes": config["names"],
    }
