"""
Supabase client for authentication only.
Uses httpx for lightweight API calls.
"""

import os
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Auth API endpoint
AUTH_URL = f"{SUPABASE_URL}/auth/v1"


def _get_headers(access_token: Optional[str] = None) -> dict:
    """Get headers for Supabase API requests."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    else:
        headers["Authorization"] = f"Bearer {SUPABASE_KEY}"
    return headers


def is_supabase_configured() -> bool:
    """Check if Supabase is properly configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


# =============================================================================
# Authentication Functions
# =============================================================================

def supabase_sign_up(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """
    Sign up a new user with Supabase Auth.
    Returns (success, message, user_data).
    """
    if not is_supabase_configured():
        return False, "Supabase not configured", None
    
    try:
        response = httpx.post(
            f"{AUTH_URL}/signup",
            headers=_get_headers(),
            json={"email": email, "password": password},
            timeout=30.0,
        )
        
        data = response.json()
        
        if response.status_code == 200:
            user = data.get("user", {})
            return True, "Registration successful! Please check your email to verify.", {
                "id": user.get("id"),
                "email": user.get("email"),
                "created_at": user.get("created_at"),
            }
        else:
            error_msg = data.get("error_description") or data.get("msg") or data.get("message") or "Registration failed"
            return False, error_msg, None
            
    except httpx.TimeoutException:
        return False, "Connection timeout. Please try again.", None
    except Exception as e:
        return False, f"Registration error: {str(e)}", None


def supabase_sign_in(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """
    Sign in a user with Supabase Auth.
    Returns (success, message, user_data).
    """
    if not is_supabase_configured():
        return False, "Supabase not configured", None
    
    try:
        response = httpx.post(
            f"{AUTH_URL}/token?grant_type=password",
            headers=_get_headers(),
            json={"email": email, "password": password},
            timeout=30.0,
        )
        
        data = response.json()
        
        if response.status_code == 200:
            user = data.get("user", {})
            return True, "Login successful", {
                "id": user.get("id"),
                "email": user.get("email"),
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token"),
            }
        else:
            error_msg = data.get("error_description") or data.get("msg") or data.get("message") or "Invalid credentials"
            return False, error_msg, None
            
    except httpx.TimeoutException:
        return False, "Connection timeout. Please try again.", None
    except Exception as e:
        return False, f"Login error: {str(e)}", None


def supabase_sign_out() -> tuple[bool, str]:
    """Sign out the current user."""
    return True, "Logged out successfully"


def supabase_get_user(access_token: str) -> Optional[dict]:
    """Get user info using access token."""
    if not is_supabase_configured() or not access_token:
        return None
    
    try:
        response = httpx.get(
            f"{AUTH_URL}/user",
            headers=_get_headers(access_token),
            timeout=30.0,
        )
        
        if response.status_code == 200:
            user = response.json()
            return {
                "id": user.get("id"),
                "email": user.get("email"),
                "created_at": user.get("created_at"),
            }
    except Exception:
        pass
    
    return None


def supabase_reset_password(email: str) -> tuple[bool, str]:
    """Send password reset email."""
    if not is_supabase_configured():
        return False, "Supabase not configured"
    
    try:
        response = httpx.post(
            f"{AUTH_URL}/recover",
            headers=_get_headers(),
            json={"email": email},
            timeout=30.0,
        )
        
        if response.status_code == 200:
            return True, "Password reset email sent. Please check your inbox."
        else:
            data = response.json()
            error_msg = data.get("error_description") or data.get("msg") or "Failed to send reset email"
            return False, error_msg
            
    except Exception as e:
        return False, f"Error: {str(e)}"
