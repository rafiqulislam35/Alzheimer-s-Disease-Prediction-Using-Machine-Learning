import sys
import types
import importlib
import pytest
from fastapi.testclient import TestClient


def import_app_with_stubs(monkeypatch):
    """
    Import app.py but stub out heavy deps (vector/ollama/joblib loading)
    so we can unit test login only.
    """

    # --- Stub vector.get_retriever() so app.py doesn't load PDF/FAISS ---
    fake_vector = types.ModuleType("vector")

    class FakeRetriever:
        def invoke(self, q):
            return []

    fake_vector.get_retriever = lambda: FakeRetriever()
    sys.modules["vector"] = fake_vector

    # --- Stub langchain OllamaLLM + prompt pipeline so `prompt | llm` works ---
    fake_langchain_ollama_llms = types.ModuleType("langchain_ollama.llms")

    class FakeOllamaLLM:
        def __init__(self, *args, **kwargs):
            pass

    fake_langchain_ollama_llms.OllamaLLM = FakeOllamaLLM
    sys.modules["langchain_ollama.llms"] = fake_langchain_ollama_llms

    fake_langchain_core_prompts = types.ModuleType("langchain_core.prompts")

    class FakeChain:
        def invoke(self, *args, **kwargs):
            return "fake_answer"

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    class FakeChatPromptTemplate:
        @staticmethod
        def from_template(t):
            return FakePrompt()

    fake_langchain_core_prompts.ChatPromptTemplate = FakeChatPromptTemplate
    sys.modules["langchain_core.prompts"] = fake_langchain_core_prompts

    # --- Stub joblib.load so model/scaler/encoder loading won't fail ---
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: object())

    # --- Ensure model files "exist" check doesn't crash ---
    from pathlib import Path

    real_exists = Path.exists

    def fake_exists(self):
        if str(self).endswith(("random_forest_model.pkl", "scaler.pkl", "label_encoder.pkl")):
            return True
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    # --- Now import app.py fresh ---
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    return app_module


@pytest.fixture
def client(monkeypatch):
    app_module = import_app_with_stubs(monkeypatch)
    return TestClient(app_module.app)


def test_login_get_returns_200(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert "Welcome back" in r.text


def test_login_post_invalid_credentials_returns_200(monkeypatch, client):
    import app as app_module

    monkeypatch.setattr(app_module.users_col, "find_one", lambda q: None)

    r = client.post("/login", data={"email": "x@test.com", "password": "wrong"})
    assert r.status_code == 200
    assert "Welcome back" in r.text


def test_login_post_disabled_user_returns_200(monkeypatch, client):
    import app as app_module

    fake_user = {
        "_id": "507f1f77bcf86cd799439011",
        "email": "u@test.com",
        "password": "hashed",
        "status": "disabled",
    }
    monkeypatch.setattr(app_module.users_col, "find_one", lambda q: fake_user)
    monkeypatch.setattr(app_module, "verify_password", lambda plain, hashed: True)

    r = client.post("/login", data={"email": "u@test.com", "password": "ok"})
    assert r.status_code == 200
    assert "Welcome back" in r.text


def test_login_post_success_user_redirects_and_sets_cookie(monkeypatch, client):
    import app as app_module

    fake_user = {
        "_id": "507f1f77bcf86cd799439011",
        "email": "u@test.com",
        "password": "hashed",
        "role": "user",
        "status": "active",
    }

    monkeypatch.setattr(app_module.users_col, "find_one", lambda q: fake_user)
    monkeypatch.setattr(app_module, "verify_password", lambda plain, hashed: True)
    monkeypatch.setattr(app_module, "is_admin_email", lambda email: False)
    monkeypatch.setattr(app_module, "create_token", lambda user_id, role="user": "TESTTOKEN")
    monkeypatch.setattr(app_module.users_col, "update_one", lambda *args, **kwargs: None)

    r = client.post("/login", data={"email": "u@test.com", "password": "ok"}, follow_redirects=False)

    assert r.status_code in (302, 303)
    assert r.headers["location"] == "/dashboard"
    assert "auth_token=" in r.headers.get("set-cookie", "")


def test_login_post_success_admin_redirects_to_admin(monkeypatch, client):
    import app as app_module

    fake_user = {
        "_id": "507f1f77bcf86cd799439011",
        "email": "admin@test.com",
        "password": "hashed",
        "role": "user",
        "status": "active",
    }

    monkeypatch.setattr(app_module.users_col, "find_one", lambda q: fake_user)
    monkeypatch.setattr(app_module, "verify_password", lambda plain, hashed: True)
    monkeypatch.setattr(app_module, "is_admin_email", lambda email: True)
    monkeypatch.setattr(app_module, "create_token", lambda user_id, role="admin": "ADMINTOKEN")
    monkeypatch.setattr(app_module.users_col, "update_one", lambda *args, **kwargs: None)

    r = client.post("/login", data={"email": "admin@test.com", "password": "ok"}, follow_redirects=False)

    assert r.status_code in (302, 303)
    assert r.headers["location"] == "/admin"
    assert "auth_token=" in r.headers.get("set-cookie", "")
