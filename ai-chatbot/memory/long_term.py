import json
import sqlite3
from datetime import datetime
import os

class LongTermMemory:
    def __init__(self, db_path='data/knowledge/memory.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence REAL,
                source TEXT,
                timestamp TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_info (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                key TEXT,
                value TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_fact(self, subject, predicate, object_, confidence=1.0, source=None):
        """Store a new fact in memory"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO facts (subject, predicate, object, confidence, source, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            subject, predicate, object_, confidence, source,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def query_facts(self, subject=None, predicate=None, object_=None):
        """Query facts from memory"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        query = "SELECT * FROM facts WHERE 1=1"
        params = []
        
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)
        if object_:
            query += " AND object = ?"
            params.append(object_)
            
        c.execute(query, params)
        results = c.fetchall()
        conn.close()
        
        return results
        
    def store_user_info(self, user_id, key, value):
        """Store user-specific information"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO user_info (user_id, key, value, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, key, value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
    def get_user_info(self, user_id, key=None):
        """Retrieve user-specific information"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if key:
            c.execute('''
                SELECT value FROM user_info
                WHERE user_id = ? AND key = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (user_id, key))
            result = c.fetchone()
        else:
            c.execute('''
                SELECT key, value FROM user_info
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id,))
            result = c.fetchall()
            
        conn.close()
        return result