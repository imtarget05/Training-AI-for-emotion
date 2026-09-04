"""
Database module: persistence for emotion recognition data.
Stores all predictions with timestamps for reporting & analytics.

Backends:
- SQLite (default, local development): file path via DB_PATH
- PostgreSQL (production free-tier hosting, e.g. Neon/Supabase):
  set DATABASE_URL (postgres://...) — schema & queries are translated
  automatically. No credentials belong in Git.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Overridable via env so tests and containers can point at their own file
DB_PATH = os.environ.get("DB_PATH", "emotion_data.db")

# Production database (PostgreSQL). Empty => local SQLite is used.
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))


class _PgCursor:
    """Translates sqlite-style '?' placeholders to psycopg2 '%s' and returns
    dual-access rows (by index or column name), mimicking sqlite3.Row."""

    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        return self._cur.execute(sql.replace("?", "%s"), params) if params is not None \
            else self._cur.execute(sql.replace("?", "%s"))

    def executemany(self, sql, seq):
        return self._cur.executemany(sql.replace("?", "%s"), seq)

    @staticmethod
    def _wrap(row):
        if row is None:
            return None
        vals = list(row.values())
        return type("_Row", (), {
            "__getitem__": lambda s, k: vals[k] if isinstance(k, int) else row[k],
        })()

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        return [self._wrap(r) for r in self._cur.fetchall()]

    def __getattr__(self, name):
        return getattr(self._cur, name)


class _PgConnection:
    """Minimal connection proxy matching the sqlite3 usage in this module."""

    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        import psycopg2.extras
        return _PgCursor(self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor))

    # sqlite3.Row support does not exist in psycopg2; queries already return
    # dual-access rows via _PgCursor, so swallow row_factory assignments.
    @property
    def row_factory(self):
        return None

    @row_factory.setter
    def row_factory(self, _):
        pass

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_connection():
    """Get a database connection (PostgreSQL if DATABASE_URL is set, else SQLite)."""
    if USE_PG:
        import psycopg2
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        # Neon/Supabase and other managed providers require SSL; local dev
        # Postgres typically doesn't. Only force sslmode=require for remote hosts.
        url = DATABASE_URL
        parsed = urlparse(url)
        if "sslmode=" not in parse_qs(parsed.query) and "sslmode" not in url:
            hostname = (parsed.hostname or "").lower()
            remote = hostname not in ("localhost", "127.0.0.1", "::1", "") and not hostname.endswith(".local")
            if remote:
                q = parsed.query + ("&" if parsed.query else "") + "sslmode=require"
                url = urlunparse(parsed._replace(query=q))
        return _PgConnection(psycopg2.connect(url))
    return sqlite3.connect(DB_PATH)


def _is_pg() -> bool:
    return USE_PG


def init_db():
    """Initialize database with required tables (SQLite or PostgreSQL)."""
    conn = get_connection()
    cursor = conn.cursor()

    pk = "SERIAL PRIMARY KEY" if _is_pg() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    ts_default = "DEFAULT CURRENT_TIMESTAMP"  # valid on both backends

    # Table: predictions - stores each emotion prediction
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id {pk},
            device_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            face_detected INTEGER DEFAULT 0,
            probs_json TEXT,
            created_at TEXT {ts_default}
        )
    """)

    # Table: sessions - track camera/device usage sessions
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS sessions (
            id {pk},
            device_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_predictions INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0.0
        )
    """)

    # Table: reports - metadata for generated reports
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS reports (
            id {pk},
            report_type TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            period_start TEXT,
            period_end TEXT,
            file_path TEXT,
            metadata_json TEXT
        )
    """)

    # Table: tutor_feedback - AI Tutor messages generated in response to
    # sustained negative emotion states (see tutor.py)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS tutor_feedback (
            id {pk},
            device_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            trigger_emotion TEXT NOT NULL,
            message TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT {ts_default}
        )
    """)


    # Indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_id ON predictions(device_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emotion ON predictions(emotion)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tutor_device_id ON tutor_feedback(device_id)")

    
    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DATABASE_URL if USE_PG else DB_PATH)


def _clean_ts(ts: str) -> str:
    """Return a timestamp string PostgreSQL can cast to DATE/TIMESTAMP."""
    if not isinstance(ts, str):
        return ts
    # '...+00:00Z' -> '...Z'  (remove redundant '+00:00' before trailing 'Z')
    if ts.endswith("+00:00Z"):
        return ts[:-7] + "Z"
    return ts


def save_prediction(device_id: str, timestamp: str, emotion: str, 
                    confidence: float, probs: Dict[str, float], 
                    face_detected: bool = True):
    """Save a single prediction to the database."""
    timestamp = _clean_ts(timestamp)
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
                _clean_ts(p["timestamp"]),
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
    
    hour_expr = (
        "to_char(timestamp::timestamp, 'YYYY-MM-DD HH24:00:00') as hour"
        if _is_pg()
        else "strftime('%Y-%m-%d %H:00:00', timestamp) as hour"
    )
    query = f"""
        SELECT
            {hour_expr},
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


def save_tutor_feedback(
    device_id: str,
    timestamp: str,
    trigger_emotion: str,
    message: str,
    source: str,
):
    """Save a generated AI Tutor message to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tutor_feedback (device_id, timestamp, trigger_emotion, message, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (device_id, _clean_ts(timestamp), trigger_emotion, message, source),
    )
    conn.commit()
    conn.close()


def get_tutor_feedback_history(
    device_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch recent AI Tutor messages, optionally filtered by device."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM tutor_feedback WHERE 1=1"
    params: List[Any] = []
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "device_id": row["device_id"],
            "timestamp": row["timestamp"],
            "trigger_emotion": row["trigger_emotion"],
            "message": row["message"],
            "source": row["source"],
            "created_at": row["created_at"],
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
