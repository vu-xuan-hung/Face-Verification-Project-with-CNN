import sqlite3
from datetime import datetime

DB_NAME = "login_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_login(username, role):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Format current time: YYYY-MM-DD HH:MM:SS
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO login_logs (username, role, timestamp)
        VALUES (?, ?, ?)
    ''', (username, role, current_time))
    conn.commit()
    conn.close()

def get_logs(username_filter=None, date_filter=None):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT username, role, timestamp FROM login_logs WHERE 1=1"
    params = []
    
    if username_filter:
        query += " AND username LIKE ?"
        params.append(f"%{username_filter}%")
        
    if date_filter:
        # Match date part of the timestamp (YYYY-MM-DD)
        query += " AND timestamp LIKE ?"
        params.append(f"{date_filter}%")
        
    query += " ORDER BY id DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
