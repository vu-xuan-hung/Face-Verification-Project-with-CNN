import sqlite3
import os
from datetime import datetime
from pathlib import Path

# Đường dẫn DB nằm ở root project
_DB_DEFAULT = Path(__file__).resolve().parents[4] / "login_logs.db"

def get_db_path():
    return str(_DB_DEFAULT)

def init_db(db_path=None):
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Bảng log đăng nhập
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # Bảng lưu thông tin user và role (CẢI TIẾN: không hardcode nữa)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')
    
    # Seed admin mặc định nếu bảng trống
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT OR IGNORE INTO users (username, role) VALUES (?, ?)", ("hung", "admin"))
        print("Đã tạo user mặc định: hung (admin)")
    
    conn.commit()
    conn.close()

def get_role(username, db_path=None):
    """Lấy role của user từ database thay vì hardcode if/else."""
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    else:
        # User chưa có trong bảng → mặc định là 'user'
        return "user"

def register_user(username, role="user", db_path=None):
    """Thêm hoặc cập nhật role của user trong database."""
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # INSERT OR REPLACE để upsert
    cursor.execute(
        "INSERT OR REPLACE INTO users (username, role) VALUES (?, ?)",
        (username, role)
    )
    conn.commit()
    conn.close()
    print(f"Đã đăng ký: {username} với role={role}")

def log_login(username, role, db_path=None):
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        'INSERT INTO login_logs (username, role, timestamp) VALUES (?, ?, ?)',
        (username, role, current_time)
    )
    conn.commit()
    conn.close()

def get_logs(username_filter=None, date_filter=None, db_path=None):
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT username, role, timestamp FROM login_logs WHERE 1=1"
    params = []
    
    if username_filter:
        query += " AND username LIKE ?"
        params.append(f"%{username_filter}%")
        
    if date_filter:
        query += " AND timestamp LIKE ?"
        params.append(f"{date_filter}%")
        
    query += " ORDER BY id DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
