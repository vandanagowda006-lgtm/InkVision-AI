import sqlite3
import bcrypt
import re

from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager


# =====================================================
# DATABASE CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATABASE_DIR / "inkvision.db"


# =====================================================
# DATABASE CONNECTION
# =====================================================

@contextmanager
def get_auth_connection():

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=10
    )

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
# INITIALIZE AUTHENTICATION DATABASE
# =====================================================

def initialize_auth_database():

    with get_auth_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                username TEXT UNIQUE NOT NULL,

                email TEXT UNIQUE NOT NULL,

                password_hash BLOB NOT NULL,

                created_at TEXT NOT NULL

            )
        """)

        # Personal profile for every registered user
        connection.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (

                user_id INTEGER PRIMARY KEY,

                display_name TEXT,

                bio TEXT DEFAULT '',

                avatar_path TEXT,

                updated_at TEXT NOT NULL,

                FOREIGN KEY(user_id)
                REFERENCES users(id)
                ON DELETE CASCADE

            )
        """)


# =====================================================
# VALIDATE USER INPUT
# =====================================================

def validate_registration(username, email, password):

    if not all(
        isinstance(value, str)
        for value in [username, email, password]
    ):
        return False, "Invalid registration details."

    username = username.strip()
    email = email.strip().lower()

    if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username):

        return False, (
            "Username must contain 3-20 letters, "
            "numbers or underscores."
        )

    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    if not re.fullmatch(email_pattern, email):

        return False, "Please enter a valid email address."

    if len(password) < 8:

        return False, "Password must contain at least 8 characters."

    if len(password.encode("utf-8")) > 72:

        return False, "Password must not exceed 72 bytes."

    return True, "Validation successful."


# =====================================================
# REGISTER NEW USER
# =====================================================

def register_user(username, email, password):

    valid, message = validate_registration(
        username,
        email,
        password
    )

    if not valid:
        return False, message

    username = username.strip()
    email = email.strip().lower()

    try:

        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12)
        )

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        with get_auth_connection() as connection:

            cursor = connection.execute("""
                INSERT INTO users
                (
                    username,
                    email,
                    password_hash,
                    created_at
                )

                VALUES (?, ?, ?, ?)
            """, (
                username,
                email,
                password_hash,
                created_at
            ))

            user_id = cursor.lastrowid

            # Automatically create a personal profile
            connection.execute("""
                INSERT INTO user_profiles
                (
                    user_id,
                    display_name,
                    bio,
                    avatar_path,
                    updated_at
                )

                VALUES (?, ?, '', NULL, ?)
            """, (
                user_id,
                username,
                created_at
            ))

        return True, (
            "Your InkVision account and personal profile "
            "were created successfully!"
        )

    except sqlite3.IntegrityError:

        return False, (
            "Username or email already exists. "
            "Please try another."
        )

    except sqlite3.Error:

        return False, "Unable to create account. Please try again."


# =====================================================
# AUTHENTICATE EXISTING USER
# =====================================================

def authenticate_user(username, password):

    if not isinstance(username, str):
        return None

    if not isinstance(password, str):
        return None

    username = username.strip()

    if not username or not password:
        return None

    if len(password.encode("utf-8")) > 72:
        return None

    try:

        with get_auth_connection() as connection:

            user = connection.execute("""
                SELECT
                    id,
                    username,
                    email,
                    password_hash,
                    created_at

                FROM users

                WHERE username = ?
            """, (username,)).fetchone()

        if user is None:
            return None

        stored_hash = bytes(user[3])

        if bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash
        ):

            return {
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "created_at": user[4]
            }

        return None

    except (sqlite3.Error, ValueError, TypeError):

        return None


# =====================================================
# CHECK USERNAME AVAILABILITY
# =====================================================

def username_exists(username):

    if not isinstance(username, str):
        return False

    try:

        with get_auth_connection() as connection:

            result = connection.execute("""
                SELECT 1

                FROM users

                WHERE username = ?
            """, (username.strip(),)).fetchone()

        return result is not None

    except sqlite3.Error:

        return False


# =====================================================
# CHECK EMAIL AVAILABILITY
# =====================================================

def email_exists(email):

    if not isinstance(email, str):
        return False

    try:

        with get_auth_connection() as connection:

            result = connection.execute("""
                SELECT 1

                FROM users

                WHERE email = ?
            """, (email.strip().lower(),)).fetchone()

        return result is not None

    except sqlite3.Error:

        return False


# =====================================================
# INITIALIZE DATABASE
# =====================================================

initialize_auth_database()
