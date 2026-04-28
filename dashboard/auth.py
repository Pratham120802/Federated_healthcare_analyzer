"""
Authentication module supporting both local and Supabase authentication.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.supabase_client import (
    is_supabase_configured,
    supabase_sign_up,
    supabase_sign_in,
    supabase_sign_out,
    supabase_get_user,
    supabase_reset_password,
)

USERS_FILE = PROJECT_ROOT / "users.json"

DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "email": "admin@example.com",
        "role": "admin"
    }
}


def _load_users() -> dict:
    """Load users from JSON file."""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_USERS.copy()


def _save_users(users: dict):
    """Save users to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_auth_mode() -> str:
    """Get current authentication mode."""
    if is_supabase_configured():
        return "supabase"
    return "local"


# ---------------------------------------------------------------------------
# Local Authentication
# ---------------------------------------------------------------------------

def check_login_local(username: str, password: str) -> bool:
    """Check local login credentials."""
    users = _load_users()
    if username in users:
        return users[username]["password_hash"] == _hash_password(password)
    return False


def register_user_local(username: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new local user."""
    users = _load_users()
    
    if username in users:
        return False, "Username already exists"
    
    if any(u.get("email") == email for u in users.values()):
        return False, "Email already registered"
    
    users[username] = {
        "password_hash": _hash_password(password),
        "email": email,
        "role": "user"
    }
    _save_users(users)
    return True, "Account created successfully! You can now sign in."


# ---------------------------------------------------------------------------
# Supabase Authentication
# ---------------------------------------------------------------------------

def check_login_supabase(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """Authenticate user with Supabase."""
    return supabase_sign_in(email, password)


def register_user_supabase(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """Register a new user with Supabase."""
    if not email or "@" not in email:
        return False, "Please enter a valid email address", None
    
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters", None
    
    return supabase_sign_up(email, password)


def reset_password(email: str) -> tuple[bool, str]:
    """Send password reset email via Supabase."""
    if not is_supabase_configured():
        return False, "Password reset requires Supabase"
    return supabase_reset_password(email)


def sign_out() -> tuple[bool, str]:
    """Sign out from Supabase."""
    return supabase_sign_out()
