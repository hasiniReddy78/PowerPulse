"""Shared layout, authentication flow, and UI components for PowerPulse."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from database.db import authenticate, create_user, init_db

THEME_CSS = """
<style>
/* Base container & typography */
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1200px;}
h1, h2, h3, h4 {letter-spacing: -0.02em; color: #3A0512;}

/* PowerPulse Maroon, Pink & White Theme */
.pp-hero {
  background: linear-gradient(135deg, #7B112D 0%, #4E0A1B 100%);
  color: #FFFFFF;
  padding: 1.5rem 1.8rem;
  border-radius: 14px;
  margin-bottom: 1.2rem;
  box-shadow: 0 6px 18px rgba(123, 17, 45, 0.15);
}
.pp-hero h1 {color: #FFFFFF; font-size: 1.95rem; margin: 0; font-weight: 700;}
.pp-kicker {color: #F8BBD0; font-size: 0.8rem; letter-spacing: 0.16em; margin: 0 0 0.4rem 0; font-weight: 600;}
.pp-hero p {color: #FCE4EC; margin: 0.5rem 0 0 0; font-size: 1.02rem;}

.pp-card {
  background: #FFFFFF;
  border: 1px solid #F3D2DC;
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  margin-bottom: 1rem;
  box-shadow: 0 2px 8px rgba(123, 17, 45, 0.04);
}

.pp-auth-card {
  background: #FFFFFF;
  border: 1.5px solid #F8BBD0;
  border-radius: 16px;
  padding: 2rem 2.2rem;
  max-width: 480px;
  margin: 1rem auto;
  box-shadow: 0 10px 30px rgba(123, 17, 45, 0.12);
}

.pp-note {
  background: #FFF0F3;
  border-left: 4px solid #7B112D;
  padding: 0.85rem 1.1rem;
  border-radius: 8px;
  margin: 0.6rem 0 1rem 0;
  color: #4A0B1A;
}

.pp-bad {
  background: #FFF1F3;
  border-left: 4px solid #C02643;
  padding: 0.85rem 1.1rem;
  border-radius: 8px;
  color: #7B0E23;
  margin: 0.5rem 0;
}

.pp-ok {
  background: #EDF9F0;
  border-left: 4px solid #1E7E34;
  padding: 0.85rem 1.1rem;
  border-radius: 8px;
  color: #155724;
  margin: 0.5rem 0;
}

.pp-badge-active {
  display: inline-block;
  background-color: #D4EDDA;
  color: #155724;
  padding: 0.25rem 0.65rem;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.82rem;
  border: 1px solid #C3E6CB;
}

.pp-badge-expired {
  display: inline-block;
  background-color: #F8D7DA;
  color: #721C24;
  padding: 0.25rem 0.65rem;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.82rem;
  border: 1px solid #F5C6CB;
}

/* Primary buttons in maroon palette */
div.stButton > button[kind="primary"] {
  background-color: #7B112D;
  color: #FFFFFF;
  border: none;
  font-weight: 600;
  border-radius: 8px;
  transition: all 0.2s ease-in-out;
}
div.stButton > button[kind="primary"]:hover {
  background-color: #9B1B3B;
  color: #FFFFFF;
  border: none;
  box-shadow: 0 4px 12px rgba(123, 17, 45, 0.25);
}

.pp-muted {color: #6C757D; font-size: 0.92rem;}
</style>
"""

HIDE_NAV_CSS = """
<style>
/* Hide sidebar navigation when user is unauthenticated */
[data-testid="stSidebarNav"] {display: none !important;}
</style>
"""


def current_user() -> dict | None:
    """Retrieve the currently authenticated service centre user from session state."""
    return st.session_state.get("user")


def hero(title: str, subtitle: str) -> None:
    """Render the standard PowerPulse hero header."""
    st.markdown(
        f"<div class='pp-hero'><p class='pp-kicker'>POWERPULSE PLATFORM</p>"
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Render a styled informational note."""
    st.markdown(f"<div class='pp-note'>{text}</div>", unsafe_allow_html=True)


def render_auth_interface(message: str | None = None) -> None:
    """Render a clean, dedicated authentication interface for Login and Sign Up."""
    if message:
        st.info(message)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 1.2rem;'>
                <h2 style='color: #7B112D; margin-bottom: 0.3rem;'>Service Centre Portal</h2>
                <p style='color: #6C757D; font-size: 0.95rem;'>Log in or register to access lithium-ion battery assessment & warranty tracking.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["🔑 Log In", "📝 Sign Up"])

        with tab_login:
            st.markdown("##### Log In to Your Account")
            login_username = st.text_input("Username", key="portal_login_user")
            login_password = st.text_input("Password", type="password", key="portal_login_pass")

            if st.button("Log In", type="primary", use_container_width=True, key="btn_login_portal"):
                clean_u = login_username.strip()
                if not clean_u or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    authenticated = authenticate(clean_u, login_password)
                    if authenticated is None:
                        st.error("Invalid username or password. Please try again.")
                    else:
                        st.session_state["user"] = authenticated
                        st.success(f"Welcome back, {authenticated['username']}!")
                        # Redirect directly to dashboard
                        try:
                            st.switch_page("pages/dashboard.py")
                        except Exception:
                            st.rerun()

        with tab_signup:
            st.markdown("##### Register New Service Centre")
            reg_username = st.text_input("Choose Username", key="portal_reg_user", help="3 to 32 characters, letters, numbers, underscore.")
            reg_password = st.text_input("Choose Password", type="password", key="portal_reg_pass", help="At least 8 characters.")
            reg_confirm = st.text_input("Confirm Password", type="password", key="portal_reg_confirm")

            if st.button("Create Account", type="primary", use_container_width=True, key="btn_reg_portal"):
                clean_u = reg_username.strip()
                if not clean_u or not reg_password:
                    st.error("Please fill in all registration fields.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    try:
                        user_id = create_user(clean_u, reg_password)
                        st.success("Account registered successfully! You can now switch to the Log In tab to access PowerPulse.")
                    except ValueError as exc:
                        st.error(str(exc))


def _render_sidebar() -> None:
    """Render the sidebar without persistent login/signup forms."""
    user = current_user()
    with st.sidebar:
        st.markdown(
            """
            <div style='padding: 0.5rem 0;'>
                <h3 style='color: #7B112D; margin: 0; font-size: 1.35rem;'>⚡ POWERPULSE</h3>
                <p style='color: #8C223D; font-size: 0.78rem; letter-spacing: 0.1em; margin: 0;'>BATTERY INTELLIGENCE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if user:
            st.markdown(
                f"""
                <div style='background: #FFF0F3; border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.6rem 0;'>
                    <span style='font-size: 0.76rem; color: #7B112D; text-transform: uppercase; font-weight: 600;'>Service Centre</span><br/>
                    <strong style='color: #3A0512; font-size: 0.95rem;'>👤 {user['username']}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🚪 Log Out", use_container_width=True, key="btn_logout_sidebar"):
                st.session_state.pop("user", None)
                st.rerun()
        else:
            st.markdown(
                """
                <div style='background: #FDF2F4; border-radius: 8px; padding: 0.65rem 0.8rem; margin: 0.6rem 0; font-size: 0.85rem; color: #7B112D;'>
                    🔒 <strong>Session Locked</strong><br/>
                    Please authenticate to access assessment tools.
                </div>
                """,
                unsafe_allow_html=True,
            )


def prepare_page(title: str, require_login: bool = True) -> None:
    """Configure page layout and gate access for unauthenticated users."""
    st.set_page_config(
        page_title=f"POWERPULSE | {title}",
        page_icon=":material/battery_charging_full:",
        layout="wide",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    init_db()

    user = current_user()
    if not user:
        # Hide default multi-page navigation when not authenticated
        st.markdown(HIDE_NAV_CSS, unsafe_allow_html=True)

    _render_sidebar()

    # Access Gate: if login is required and user is not authenticated, render auth interface and stop
    if require_login and user is None:
        hero(
            "Service Centre Authentication",
            "Access to diagnostic telemetry, health assessment, and warranty tracking requires an authenticated session.",
        )
        render_auth_interface()
        st.stop()
