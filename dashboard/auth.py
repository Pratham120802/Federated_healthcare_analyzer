"""
Authentication module with user registration and Google OAuth support.
"""

import json
import hashlib
import secrets
from pathlib import Path
from typing import Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USERS_FILE = PROJECT_ROOT / "users.json"

# Default users (created on first run)
DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "email": "admin@example.com",
        "created_at": "2024-01-01T00:00:00",
        "auth_type": "local"
    },
    "doctor": {
        "password_hash": hashlib.sha256("health2024".encode()).hexdigest(),
        "email": "doctor@example.com",
        "created_at": "2024-01-01T00:00:00",
        "auth_type": "local"
    },
    "researcher": {
        "password_hash": hashlib.sha256("fedlearn".encode()).hexdigest(),
        "email": "researcher@example.com",
        "created_at": "2024-01-01T00:00:00",
        "auth_type": "local"
    }
}


def _load_users() -> dict:
    """Load users from JSON file."""
    if not USERS_FILE.exists():
        _save_users(DEFAULT_USERS)
        return DEFAULT_USERS.copy()
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return DEFAULT_USERS.copy()


def _save_users(users: dict):
    """Save users to JSON file."""
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_login(username: str, password: str) -> bool:
    """Validate username and password."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return False
    return user.get("password_hash") == _hash_password(password)


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message).
    """
    users = _load_users()
    
    # Validate username
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters"
    if not username.isalnum():
        return False, "Username must be alphanumeric"
    if username.lower() in [u.lower() for u in users.keys()]:
        return False, "Username already exists"
    
    # Validate email
    if not email or "@" not in email:
        return False, "Invalid email address"
    if any(u.get("email", "").lower() == email.lower() for u in users.values()):
        return False, "Email already registered"
    
    # Validate password
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    # Create user
    users[username] = {
        "password_hash": _hash_password(password),
        "email": email.lower(),
        "created_at": datetime.now().isoformat(),
        "auth_type": "local"
    }
    _save_users(users)
    
    return True, "Registration successful! You can now login."


def register_google_user(email: str, name: str, google_id: str) -> tuple[bool, str]:
    """
    Register or login a Google OAuth user.
    Returns (success, username).
    """
    users = _load_users()
    
    # Check if user exists by Google ID or email
    for username, user_data in users.items():
        if user_data.get("google_id") == google_id:
            return True, username
        if user_data.get("email", "").lower() == email.lower() and user_data.get("auth_type") == "google":
            return True, username
    
    # Create new username from email
    base_username = email.split("@")[0].replace(".", "").replace("-", "")[:15]
    username = base_username
    counter = 1
    while username.lower() in [u.lower() for u in users.keys()]:
        username = f"{base_username}{counter}"
        counter += 1
    
    # Create user
    users[username] = {
        "password_hash": "",
        "email": email.lower(),
        "name": name,
        "google_id": google_id,
        "created_at": datetime.now().isoformat(),
        "auth_type": "google"
    }
    _save_users(users)
    
    return True, username


def get_user_info(username: str) -> Optional[dict]:
    """Get user information."""
    users = _load_users()
    user = users.get(username)
    if user:
        return {
            "username": username,
            "email": user.get("email", ""),
            "name": user.get("name", username),
            "auth_type": user.get("auth_type", "local"),
            "created_at": user.get("created_at", "")
        }
    return None


def get_all_usernames() -> list[str]:
    """Get list of all registered usernames."""
    users = _load_users()
    return list(users.keys())
