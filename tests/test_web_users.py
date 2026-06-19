import io
import json

import telegram_notebook.main as web
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.webauth import hash_password, verify_password, web_owner_id

SINGLE_CHAT = {
    "name": "Alice Chat",
    "type": "personal_chat",
    "id": 7,
    "messages": [
        {"id": 1, "type": "message", "date": "2021-01-01T00:00:00", "from": "A", "text": "secret alpha note"},
        {"id": 2, "type": "message", "date": "2021-01-01T00:01:00", "from": "A", "text": "beta"},
    ],
}


# --- webauth pure ----------------------------------------------------------

def test_hash_verify_roundtrip():
    encoded = hash_password("s3cret!")
    assert verify_password("s3cret!", encoded)
    assert not verify_password("wrong", encoded)


def test_hash_password_is_salted():
    assert hash_password("same") != hash_password("same")  # random salt


def test_verify_password_rejects_garbage():
    assert not verify_password("x", "not-an-encoded-hash")


def test_web_owner_id_is_negative_namespace():
    assert web_owner_id(1) == -1
    assert web_owner_id(42) == -42


# --- repository: web users & sessions --------------------------------------

def test_create_web_user_is_unique(tmp_path):
    repo = Repository(tmp_path / "s.db")
    repo.init()
    uid = repo.create_web_user(username="alice", password_hash="h")
    assert uid
    assert repo.create_web_user(username="alice", password_hash="h2") is None
    assert repo.get_web_user_by_name(username="alice")["id"] == uid


def test_web_session_lifecycle(tmp_path):
    repo = Repository(tmp_path / "s.db")
    repo.init()
    uid = repo.create_web_user(username="bob", password_hash="h")
    repo.create_web_session(token="tok", web_user_id=uid)
    user = repo.get_web_session_user(token="tok")
    assert user["username"] == "bob" and user["id"] == uid
    repo.delete_web_session(token="tok")
    assert repo.get_web_session_user(token="tok") is None
    assert repo.get_web_session_user(token="") is None


# --- HTTP auth flow --------------------------------------------------------

class FakeReq(web.RequestHandler):
    def __init__(self, path, *, body=b"", cookie=None, client="127.0.0.1"):
        self.path = path
        self.rfile = io.BytesIO(body)
        self.headers = {}
        if body:
            self.headers["Content-Length"] = str(len(body))
        if cookie:
            self.headers["Cookie"] = cookie
        self.client_address = (client, 1)
        self.sent = None
        self.set_cookie = None

    def _send_json(self, payload, status=200, extra_headers=None):
        self.sent = (status, payload)
        for key, value in extra_headers or []:
            if key == "Set-Cookie":
                self.set_cookie = value


def _wire(monkeypatch, tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    monkeypatch.setattr(web.state, "repository", repo)
    monkeypatch.setattr(web.state.settings, "web_api_token", None)
    pipeline = IngestionPipeline(
        settings=web.state.settings,
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="m"),
    )
    monkeypatch.setattr(web.state, "pipeline", pipeline)
    return repo


def _post(path, payload, cookie=None):
    return FakeReq(path, body=json.dumps(payload).encode("utf-8"), cookie=cookie)


def test_register_sets_cookie_and_me_resolves(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    reg = _post("/api/auth/register", {"username": "alice", "password": "password1"})
    reg.do_POST()
    assert reg.sent[0] == 200 and reg.sent[1]["user"]["username"] == "alice"
    assert reg.set_cookie and reg.set_cookie.startswith("tgnb_session=")

    cookie = reg.set_cookie.split(";")[0]
    me = FakeReq("/api/auth/me", cookie=cookie)
    me.do_GET()
    assert me.sent[1]["user"]["username"] == "alice"

    # No cookie -> anonymous.
    anon = FakeReq("/api/auth/me")
    anon.do_GET()
    assert anon.sent[1]["user"] is None


def test_register_rejects_duplicate_and_short(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    _post("/api/auth/register", {"username": "alice", "password": "password1"}).do_POST()

    dup = _post("/api/auth/register", {"username": "alice", "password": "password2"})
    dup.do_POST()
    assert dup.sent[0] == 409

    short = _post("/api/auth/register", {"username": "ab", "password": "x"})
    short.do_POST()
    assert short.sent[0] == 400


def test_login_wrong_password_is_401(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    _post("/api/auth/register", {"username": "alice", "password": "password1"}).do_POST()
    bad = _post("/api/auth/login", {"username": "alice", "password": "nope"})
    bad.do_POST()
    assert bad.sent[0] == 401


def test_logout_clears_session(monkeypatch, tmp_path):
    repo = _wire(monkeypatch, tmp_path)
    reg = _post("/api/auth/register", {"username": "alice", "password": "password1"})
    reg.do_POST()
    cookie = reg.set_cookie.split(";")[0]
    token = cookie.split("=", 1)[1]
    assert repo.get_web_session_user(token=token) is not None

    out = FakeReq("/api/auth/logout", cookie=cookie)
    out.do_POST()
    assert out.sent[0] == 200
    assert repo.get_web_session_user(token=token) is None


def test_logged_in_archive_is_isolated_from_shared(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    reg = _post("/api/auth/register", {"username": "alice", "password": "password1"})
    reg.do_POST()
    cookie = reg.set_cookie.split(";")[0]

    imp = FakeReq("/api/backup/import", body=json.dumps(SINGLE_CHAT).encode("utf-8"), cookie=cookie)
    imp.headers["X-Filename"] = "result.json"
    imp.do_POST()
    assert imp.sent[0] == 200 and imp.sent[1]["messages"] == 2

    # Alice sees her own items.
    mine = FakeReq("/api/stats", cookie=cookie)
    mine.do_GET()
    assert mine.sent[1]["items"] == 2

    # The shared workspace (no login) stays empty.
    shared = FakeReq("/api/stats")
    shared.do_GET()
    assert shared.sent[1]["items"] == 0
