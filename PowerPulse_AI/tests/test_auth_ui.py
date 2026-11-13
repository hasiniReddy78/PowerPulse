"""Automated tests for authentication flow, session gating, and UI navigation."""

import pytest
from streamlit.testing.v1 import AppTest

from database.db import authenticate, create_user, hash_password, verify_password
from ui.common import HIDE_NAV_CSS, current_user


def test_password_hashing_and_verification():
    """Verify PBKDF2 hashing preserves security and round-trips correctly."""
    plain = "ServiceCentrePass123"
    hashed = hash_password(plain)
    assert plain not in hashed
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_user_creation_and_authentication(tmp_path):
    """Verify registration and authentication workflows with database isolation."""
    db = tmp_path / "auth_test.db"
    user_id = create_user("service_sc1", "WorkshopKey@99", path=db)
    assert user_id > 0

    # Successful login
    auth = authenticate("service_sc1", "WorkshopKey@99", path=db)
    assert auth is not None
    assert auth["id"] == user_id
    assert auth["username"] == "service_sc1"

    # Failed login with wrong password
    assert authenticate("service_sc1", "InvalidPass", path=db) is None

    # Duplicate username blocked
    with pytest.raises(ValueError, match="already registered"):
        create_user("service_sc1", "AnotherPassword123", path=db)


def test_unauthenticated_app_hides_sidebar_nav_and_shows_auth():
    """Verify app.py hides navigation and renders login portal when unauthenticated."""
    at = AppTest.from_file("app.py")
    at.run(timeout=15)
    assert not at.exception

    # Unauthenticated session must have no user
    assert "user" not in at.session_state or at.session_state["user"] is None

    # Main page renders authentication portal components
    markdown_texts = [m.value for m in at.markdown]
    assert any("Service Centre Portal" in txt or "Welcome to PowerPulse" in txt for txt in markdown_texts)

    # Sidebar does NOT contain radio button for persistent login/signup
    radio_labels = [r.label for r in at.sidebar.radio]
    assert "Account action" not in radio_labels


def test_protected_pages_blocked_for_unauthenticated_users():
    """Verify that pages/ cannot be accessed directly without an authenticated session."""
    protected_pages = [
        "pages/battery_assessment.py",
        "pages/dashboard.py",
        "pages/history.py",
        "pages/reports.py",
    ]

    for page_path in protected_pages:
        at = AppTest.from_file(page_path)
        at.run(timeout=15)
        # Page should stop cleanly at the auth gate without runtime exceptions
        assert not at.exception, f"Exception occurred in {page_path}: {at.exception}"

        # Markdown shows authentication prompt
        markdown_texts = [m.value for m in at.markdown]
        assert any("Service Centre Portal" in txt or "Authentication Required" in txt or "Service Centre Authentication" in txt for txt in markdown_texts), (
            f"Expected auth gate in {page_path}"
        )


def test_authenticated_user_accesses_protected_page():
    """Verify that setting authenticated user session unlocks protected page."""
    at = AppTest.from_file("pages/dashboard.py")
    # Simulate active login in session state
    at.session_state["user"] = {"id": 1, "username": "active_technician"}
    at.run(timeout=15)
    assert not at.exception

    # Sidebar shows username
    sidebar_texts = [m.value for m in at.sidebar.markdown]
    assert any("active_technician" in txt for txt in sidebar_texts)

    # Dashboard metrics are rendered
    assert len(at.metric) >= 4

