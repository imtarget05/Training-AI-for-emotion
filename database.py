"""
Database module: SQLite persistence for emotion recognition data.
Stores all predictions with timestamps for reporting & analytics.
"""
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

DB_PATH = "emotion_data.db"


def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table: predictions - stores each emotion prediction
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            face_detected INTEGER DEFAULT 0,
            probs_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table: sessions - track camera/device usage sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_predictions INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0.0
        )
    """)
    
    # Table: reports - metadata for generated reports
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            period_start TEXT,
            period_end TEXT,
            file_path TEXT,
            metadata_json TEXT
        )
    """)
    
    # Indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_id ON predictions(device_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emotion ON predictions(emotion)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DB_PATH)


def get_connection():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def save_prediction(device_id: str, timestamp: str, emotion: str, 
                    confidence: float, probs: Dict[str, float], 
                    face_detected: bool = True):
    """Save a single prediction to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (device_id, timestamp, emotion, confidence, 
                                 face_detected, probs_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            device_id,
            timestamp,
            emotion,
            confidence,
            1 if face_detected else 0,
            json.dumps(probs) if probs else None
        )
    )
    conn.commit()
    conn.close()


def save_batch_predictions(predictions: List[Dict[str, Any]]):
    """Save multiple predictions in one transaction for performance."""
    if not predictions:
        return 0
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO predictions (device_id, timestamp, emotion, confidence, 
                                 face_detected, probs_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                p["device_id"],
                p["timestamp"],
                p["emotion"],
                p["confidence"],
                1 if p.get("face_detected", True) else 0,
                json.dumps(p.get("probs", {})) if p.get("probs") else None
            )
            for p in predictions
        ]
    )
    conn.commit()
    rowcount = cursor.rowcount
    conn.close()
    return rowcount


def get_predictions(
    device_id: Optional[str] = None,
    emotion: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Query predictions with optional filters.
    Returns list of prediction dicts.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    if emotion:
        query += " AND emotion = ?"
        params.append(emotion)
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "device_id": row["device_id"],
            "timestamp": row["timestamp"],
            "emotion": row["emotion"],
            "confidence": row["confidence"],
            "face_detected": bool(row["face_detected"]),
            "probs": json.loads(row["probs_json"]) if row["probs_json"] else {},
            "created_at": row["created_at"]
        })
    
    return results


def get_emotion_distribution(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    device_id: Optional[str] = None
) -> Dict[str, int]:
    """Get count of each emotion type in the dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT emotion, COUNT(*) as count FROM predictions WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    
    query += " GROUP BY emotion"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return {row[0]: row[1] for row in rows}


def get_confidence_statistics(
    emotion: Optional[str] = None,
    device_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, float]:
    """Get avg, min, max confidence statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM predictions WHERE 1=1"
    params = []
    
    if emotion:
        query += " AND emotion = ?"
        params.append(emotion)
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0] is not None:
        return {
            "avg_confidence": round(row[0], 4),
            "min_confidence": round(row[1], 4),
            "max_confidence": round(row[2], 4)
        }
    return {"avg_confidence": 0.0, "min_confidence": 0.0, "max_confidence": 0.0}


def get_predictions_per_hour(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    device_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get predictions grouped by hour for time-series analysis."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM predictions 
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    
    query += " GROUP BY hour ORDER BY hour"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "hour": row[0],
            "count": row[1],
            "avg_confidence": round(row[2], 4) if row[2] else 0.0
        }
        for row in rows
    ]


def get_daily_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    device_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get daily aggregated statistics for reports."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total_predictions,
            COUNT(DISTINCT emotion) as unique_emotions,
            AVG(confidence) as avg_confidence,
            MIN(timestamp) as first_prediction,
            MAX(timestamp) as last_prediction
        FROM predictions 
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    
    query += " GROUP BY date ORDER BY date DESC LIMIT 30"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "date": row[0],
            "total_predictions": row[1],
            "unique_emotions": row[2],
            "avg_confidence": round(row[3], 4) if row[3] else 0.0,
            "first_prediction": row[4],
            "last_prediction": row[5]
        }
        for row in rows
    ]


def get_device_stats(device_id: Optional[str] = None) -> Dict[str, Any]:
    """Get overall statistics per device."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if device_id:
        query = """
            SELECT 
                device_id,
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                COUNT(DISTINCT emotion) as emotion_diversity,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM predictions 
            WHERE device_id = ?
            GROUP BY device_id
        """
        cursor.execute(query, (device_id,))
    else:
        query = """
            SELECT 
                device_id,
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                COUNT(DISTINCT emotion) as emotion_diversity,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM predictions 
            GROUP BY device_id
            ORDER BY total_predictions DESC
        """
        cursor.execute(query)
    
    rows = cursor.fetchall()
    conn.close()
    
    stats = {}
    for row in rows:
        stats[row[0]] = {
            "total_predictions": row[1],
            "avg_confidence": round(row[2], 4) if row[2] else 0.0,
            "emotion_diversity": row[3],
            "first_seen": row[4],
            "last_seen": row[5]
        }
    
    return stats
