"""Password hashing and session helpers for the multi-user web app.

Pure standard-library crypto (PBKDF2-HMAC-SHA256), kept separate from the HTTP
layer so it is trivially unit-testable. Web users live in their own ``owner_id``
namespace — negative ids, derived from the web-user row id — so their archives
never collide with the bot's per-Telegram-user archives (positive ids) or the
shared web workspace (``owner_id = 0``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 200_000


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def hash_password(password: str, *, iterations: int = _ITERATIONS, salt: bytes | None = None) -> str:
    """Return an encoded ``algo$iterations$salt$hash`` string for ``password``."""
    if not password:
        raise ValueError("password must not be empty")
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGO}${iterations}${_b64(salt)}${_b64(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against an encoded hash."""
    try:
        algo, iterations, salt_b64, hash_b64 = encoded.split("$")
        if algo != _ALGO:
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(hash_b64)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(derived, expected)
    except (ValueError, TypeError, AttributeError):
        return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def web_owner_id(web_user_id: int) -> int:
    """Map a web-user row id to its archive owner id (a negative namespace)."""
    return -int(web_user_id)
