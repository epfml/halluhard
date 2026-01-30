"""
Global state and session management.
"""

import hashlib
import secrets
import time


# Session duration (24 hours)
SESSION_DURATION = 24 * 60 * 60

# Global state (shared across all users)
STATE = {
    # Server configuration (shared)
    "server_mode": False,
    "data_dir": "",
    # Preloaded file (for local mode or when file specified on command line)
    "preload_input": None,     # Input file path to preload
    "preload_output": None,    # Output file path to preload
    # Authentication (shared)
    "password_hash": None,     # SHA256 hash of password (None = no auth required)
    "sessions": {},            # token -> UserSession mapping
    "failed_attempts": {},     # ip -> (count, last_attempt_time)
}


class UserSession:
    """Per-user session state for multi-user support."""
    
    def __init__(self, token):
        self.token = token
        self.expiry = time.time() + SESSION_DURATION
        # User identification
        self.username = ""
        # File state (per user)
        self.input_file = ""
        self.output_path = ""
        self.file_loaded = False
        # Data state (per user)
        self.conversations = []
        self.assistant_turns = []
        self.extracted_claims = {}  # Map: "conv_idx:turn_num" -> list of claims
        # Navigation state (per user)
        self.current_flat_index = 0
    
    def is_expired(self):
        return time.time() > self.expiry
    
    def refresh(self):
        """Refresh session expiry."""
        self.expiry = time.time() + SESSION_DURATION


def hash_password(password):
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def generate_session_token():
    """Generate a secure random session token."""
    return secrets.token_urlsafe(32)


def create_session():
    """Create a new authenticated session and return the token."""
    from .data import load_file_for_annotation
    
    token = generate_session_token()
    session = UserSession(token)
    STATE["sessions"][token] = session
    
    # If there's a preloaded file, load it into this session
    if STATE["preload_input"] and STATE["preload_output"]:
        load_file_for_annotation(session, STATE["preload_input"], STATE["preload_output"])
    
    return token


def get_session(token):
    """Get session by token, or None if invalid/expired."""
    if not token or token not in STATE["sessions"]:
        return None
    session = STATE["sessions"][token]
    if session.is_expired():
        del STATE["sessions"][token]
        return None
    return session


def is_session_valid(token):
    """Check if a session token is valid and not expired."""
    return get_session(token) is not None


def get_session_token_from_cookie(cookie_header):
    """Extract session token from Cookie header."""
    if not cookie_header:
        return None
    for part in cookie_header.split(';'):
        part = part.strip()
        if part.startswith('session='):
            return part[8:]
    return None


def check_failed_attempt_lockout(client_ip):
    """Check if client IP is locked out due to failed attempts. Returns seconds to wait or 0."""
    if client_ip not in STATE["failed_attempts"]:
        return 0
    count, last_time = STATE["failed_attempts"][client_ip]
    if count >= 3:
        elapsed = time.time() - last_time
        if elapsed < 5:
            return 5 - elapsed
    return 0


def record_failed_attempt(client_ip):
    """Record a failed login attempt."""
    now = time.time()
    if client_ip in STATE["failed_attempts"]:
        count, _ = STATE["failed_attempts"][client_ip]
        STATE["failed_attempts"][client_ip] = (count + 1, now)
    else:
        STATE["failed_attempts"][client_ip] = (1, now)


def clear_failed_attempts(client_ip):
    """Clear failed attempts after successful login."""
    if client_ip in STATE["failed_attempts"]:
        del STATE["failed_attempts"][client_ip]

