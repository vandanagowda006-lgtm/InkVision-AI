
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime


# =====================================================
# DATABASE CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATABASE_DIR / "inkvision.db"


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_connection():

    connection = sqlite3.connect(str(DB_PATH), timeout=10)

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# =====================================================
# DATABASE INITIALIZATION
# =====================================================

def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                display_name TEXT,
                bio TEXT DEFAULT '',
                avatar_path TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                predicted_digit INTEGER NOT NULL,
                confidence REAL NOT NULL,
                input_type TEXT NOT NULL,
                actual_digit INTEGER,
                feedback_note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE
            )
        """)

        # Safely migrate older prediction tables
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        if "actual_digit" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN actual_digit INTEGER
            """)

        if "feedback_note" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN feedback_note TEXT
            """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS challenge_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_user
            ON predictions(user_id, id DESC)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_challenges_user
            ON challenge_scores(user_id, id DESC)
        """)


# =====================================================
# PROFILE MANAGEMENT
# =====================================================

def create_user_profile(user_id, display_name=None):

    with get_connection() as connection:

        connection.execute("""
            INSERT OR IGNORE INTO user_profiles
            (user_id, display_name, bio, avatar_path, updated_at)
            VALUES (?, ?, '', NULL, ?)
        """, (
            user_id,
            display_name,
            current_timestamp()
        ))


def get_user_profile(user_id):

    with get_connection() as connection:

        profile = connection.execute("""
            SELECT user_id, display_name, bio,
                   avatar_path, updated_at
            FROM user_profiles
            WHERE user_id = ?
        """, (user_id,)).fetchone()

        return dict(profile) if profile else None


def update_user_profile(
    user_id,
    display_name,
    bio="",
    avatar_path=None
):

    display_name = (display_name or "").strip()
    bio = (bio or "").strip()

    if len(display_name) > 50 or len(bio) > 250:
        return False

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO user_profiles
            (user_id, display_name, bio, avatar_path, updated_at)
            VALUES (?, ?, ?, ?, ?)

            ON CONFLICT(user_id)
            DO UPDATE SET
                display_name = excluded.display_name,
                bio = excluded.bio,
                avatar_path = excluded.avatar_path,
                updated_at = excluded.updated_at
        """, (
            user_id,
            display_name,
            bio,
            avatar_path,
            current_timestamp()
        ))

    return True


# =====================================================
# SAVE PREDICTIONS
# =====================================================

def save_prediction(user_id, digit, confidence, input_type):

    digit = int(digit)
    confidence = float(confidence)

    if not 0 <= digit <= 9:
        raise ValueError("Predicted digit must be between 0 and 9.")

    if not 0 <= confidence <= 100:
        raise ValueError("Invalid confidence value.")

    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO predictions
            (user_id, predicted_digit, confidence,
             input_type, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            digit,
            confidence,
            str(input_type),
            current_timestamp()
        ))

        return cursor.lastrowid


# =====================================================
# PREDICTION HISTORY
# =====================================================

def get_user_history(user_id):

    with get_connection() as connection:

        history = connection.execute("""
            SELECT
                id,
                predicted_digit,
                confidence,
                input_type,
                created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()

        # Return tuples to match the existing app.py DataFrame
        return [
            (
                row["id"],
                row["predicted_digit"],
                row["confidence"],
                row["input_type"],
                row["created_at"]
            )
            for row in history
        ]


# =====================================================
# USER FEEDBACK
# =====================================================

def save_prediction_feedback(
    user_id,
    prediction_id,
    actual_digit,
    feedback_note=""
):

    actual_digit = int(actual_digit)

    if not 0 <= actual_digit <= 9:
        return False

    with get_connection() as connection:

        cursor = connection.execute("""
            UPDATE predictions
            SET actual_digit = ?,
                feedback_note = ?
            WHERE id = ? AND user_id = ?
        """, (
            actual_digit,
            str(feedback_note)[:250],
            prediction_id,
            user_id
        ))

        return cursor.rowcount > 0


# =====================================================
# USER STATISTICS
# =====================================================

def get_user_statistics(user_id):

    with get_connection() as connection:

        statistics = connection.execute("""
            SELECT
                COUNT(*) AS total_predictions,
                AVG(confidence) AS average_confidence,
                COUNT(DISTINCT predicted_digit) AS unique_digits
            FROM predictions
            WHERE user_id = ?
        """, (user_id,)).fetchone()

        favourite = connection.execute("""
            SELECT predicted_digit, COUNT(*) AS frequency
            FROM predictions
            WHERE user_id = ?
            GROUP BY predicted_digit
            ORDER BY frequency DESC, predicted_digit ASC
            LIMIT 1
        """, (user_id,)).fetchone()

        return {
            "total_predictions": statistics["total_predictions"] or 0,
            "average_confidence": round(
                statistics["average_confidence"] or 0, 2
            ),
            "unique_digits": statistics["unique_digits"] or 0,
            "favourite_digit": (
                favourite["predicted_digit"] if favourite else None
            )
        }


# =====================================================
# INKBATTLE CHALLENGE SCORES
# =====================================================

def save_challenge_score(user_id, score, total_questions):

    score = int(score)
    total_questions = int(total_questions)

    if total_questions <= 0:
        raise ValueError("Total questions must be greater than zero.")

    if not 0 <= score <= total_questions:
        raise ValueError("Invalid challenge score.")

    accuracy = (score / total_questions) * 100

    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO challenge_scores
            (user_id, score, total_questions, accuracy, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            score,
            total_questions,
            round(accuracy, 2),
            current_timestamp()
        ))

        return cursor.lastrowid


def get_user_challenge_history(user_id):

    with get_connection() as connection:

        results = connection.execute("""
            SELECT id, score, total_questions, accuracy, created_at
            FROM challenge_scores
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()

        return [dict(row) for row in results]


# =====================================================
# DELETE USER PREDICTION HISTORY
# =====================================================

def delete_user_history(user_id):

    with get_connection() as connection:

        connection.execute("""
            DELETE FROM predictions
            WHERE user_id = ?
        """, (user_id,))


# =====================================================
# INITIALIZE DATABASE
# =====================================================

initialize_database()
