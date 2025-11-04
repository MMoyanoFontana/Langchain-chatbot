# src/auth.py
import os
import sqlite3
from pathlib import Path
from typing import Tuple
import bcrypt

DB_PATH = Path(os.getenv("APP_DB_PATH", "app.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def verify(username: str, password: str) -> Tuple[bool, str]:
    with _conn() as c:
        row = c.execute(
            "SELECT hash_password FROM users WHERE username=?", (username,)
        ).fetchone()
    if row is None:
        return False, "Usuario o contraseña inválidos."
    pw_hash = row["hash_password"]
    ok = bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    return (ok, "OK" if ok else "Usuario o contraseña inválidos.")


def _auth(username: str, password: str) -> bool:
    ok, _ = verify(username, password)
    return ok
