# core/database.py
import sqlite3
import json
from datetime import datetime
from config import DATABASE_CONFIG

class DetectionDB:
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_CONFIG['detection_log']
        self.init_database()
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                image_data BLOB,
                camera_id TEXT DEFAULT 'Hikvision_001'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                department TEXT,
                position TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_stats (
                date DATE PRIMARY KEY,
                total_detections INTEGER DEFAULT 0,
                unique_persons INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_detection(self, person_name, confidence=None, image_data=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (person_name, confidence, image_data)
            VALUES (?, ?, ?)
        ''', (person_name, confidence, image_data))
        
        # Update daily stats
        today = datetime.now().date()
        cursor.execute('''
            INSERT OR REPLACE INTO detection_stats (date, total_detections, unique_persons)
            VALUES (?, 
                    COALESCE((SELECT total_detections FROM detection_stats WHERE date = ?), 0) + 1,
                    CASE WHEN ? NOT IN (
                        SELECT DISTINCT person_name FROM detections 
                        WHERE DATE(detection_time) = ? AND person_name != 'Unknown'
                    ) THEN COALESCE((SELECT unique_persons FROM detection_stats WHERE date = ?), 0) + 1
                    ELSE COALESCE((SELECT unique_persons FROM detection_stats WHERE date = ?), 0)
                    END
                   )
        ''', (today, today, person_name, today, today, today))
        
        conn.commit()
        conn.close()
        
    def get_recent_detections(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT person_name, detection_time, confidence 
            FROM detections 
            ORDER BY detection_time DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'name': row[0],
                'time': row[1],
                'confidence': row[2]
            }
            for row in results
        ]
        
    def get_detection_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM detections 
            WHERE DATE(detection_time) = DATE('now')
        ''')
        today_count = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(DISTINCT person_name) FROM detections 
            WHERE DATE(detection_time) = DATE('now') AND person_name != 'Unknown'
        ''')
        unique_today = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'today_detections': today_count,
            'unique_today': unique_today,
            'total_detections': total_count
        }