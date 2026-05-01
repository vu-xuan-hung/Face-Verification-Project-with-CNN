import numpy as np
import os
from vshield.core.embedder import img_to_encoding_file

def load_database(faces_dir="data/faces"):
    database_faces = {}
    print("Loading face dataset...")
    
    if not os.path.exists(faces_dir):
        print(f"Warning: Folder {faces_dir} không tồn tại!")
        return database_faces

    for username in os.listdir(faces_dir):
        user_folder = os.path.join(faces_dir, username)
        if os.path.isdir(user_folder):
            embeddings = []
            for img_name in os.listdir(user_folder):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(user_folder, img_name)
                    try:
                        emb = img_to_encoding_file(img_path)
                        embeddings.append(emb)
                    except Exception as e:
                        print(f"Lỗi đọc ảnh {img_path}: {e}")
            
            if embeddings:
                database_faces[username] = embeddings
                
    print("Dataset loaded successfully!")
    return database_faces


# ------------------------------------------------------------------
# FAISS-accelerated search (CẢI TIẾN #1)
# Nếu máy chưa cài faiss thì tự động fallback về for-loop cũ
# Cài: pip install faiss-cpu   hoặc thêm vào pyproject.toml
# ------------------------------------------------------------------

def build_faiss_index(database_faces):
    """Xây dựng FAISS index từ dictionary embeddings.
    
    Trả về (index, id_to_name) hoặc (None, None) nếu faiss chưa được cài.
    """
    try:
        import faiss

        all_embeddings = []
        id_to_name = {}  # ánh xạ: index số nguyên → username
        idx = 0

        for name, emb_list in database_faces.items():
            for emb in emb_list:
                all_embeddings.append(emb.astype(np.float32))
                id_to_name[idx] = name
                idx += 1

        if not all_embeddings:
            return None, None

        dim = len(all_embeddings[0])
        matrix = np.stack(all_embeddings)

        # Flat L2 index — chính xác 100%, tốc độ O(1) thay vì O(n)
        index = faiss.IndexFlatL2(dim)
        index.add(matrix)

        print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
        return index, id_to_name

    except ImportError:
        print("faiss chưa được cài — dùng for-loop thay thế (chạy vẫn đúng, chỉ chậm hơn)")
        return None, None


def who_is_it(encoding, database_faces, threshold=0.9, faiss_index=None, id_to_name=None):
    """Nhận diện danh tính.
    
    Nếu có faiss_index thì dùng FAISS (nhanh), không thì dùng for-loop (đúng như cũ).
    """

    # --- Cách 1: FAISS (nhanh, dùng khi đã build index) ---
    if faiss_index is not None and id_to_name is not None:
        try:
            import faiss
            query = np.expand_dims(encoding.astype(np.float32), axis=0)
            distances, indices = faiss_index.search(query, k=1)
            min_dist = float(distances[0][0]) ** 0.5  # FAISS trả squared L2
            best_idx = int(indices[0][0])
            identity = id_to_name.get(best_idx, "Unknown")
            
            if min_dist > threshold:
                return "Unknown"
            return identity
        except Exception as e:
            print(f"FAISS search lỗi, fallback to for-loop: {e}")

    # --- Cách 2: For-loop (giống code gốc, luôn hoạt động) ---
    min_dist = 100
    identity = "Unknown"
    
    for name, embeddings_list in database_faces.items():
        for db_emb in embeddings_list:
            dist = np.linalg.norm(encoding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name
                
    if min_dist > threshold: 
        return "Unknown"
    else:
        return identity
