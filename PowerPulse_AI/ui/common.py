"""Shared layout for the Streamlit pages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from database.db import authenticate, create_user, init_db

CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.pp-hero {
  background: #0B3A5B;
  color: #F4F7FA;
  padding: 1.35rem 1.5rem;
  border-radius: 12px;
  margin-bottom: 1rem;
}
.pp-hero h1 {color: #F4F7FA; font-size: 1.85rem; margin: 0;}
.pp-kicker {color: #F0C14D; font-size: 0.78rem; letter-spacing: 0.14em; margin: 0 0 0.35rem 0;}
.pp-hero p {color: #D5E3EE; margin: 0.45rem 0 0 0;}
.pp-card {
  background: #FFFFFF;
  border: 1px solid #E1E7EE;
  border-radius: 10px;
  padding: 0.9rem 1rem 0.2rem 1rem;
  margin-bottom: 0.75rem;
}
.pp-note {
  background: #F7F1E4;
  border-left: 4px solid #A56B00;
  padding: 0.75rem 0.9rem;
  border-radius: 6px;
  margin: 0.4rem 0 0.8rem 0;
}
.pp-bad {
  background: #FDECEC;
  border-left: 4px solid #A32020;
  padding: 0.75rem 0.9rem;
  border-radius: 6px;
}
.pp-ok {
  background: #EEF8F0;
  border-left: 4px solid #1F7A3A;
  padding: 0.75rem 0.9rem;
  border-radius: 6px;
}
.pp-muted {color: #526272; font-size: 0.92rem;}
</style>
"""


def prepare_page(title: str) -> None:
    st.set_page_config(
        page_title=f"POWERPULSE | {title}",
        page_icon=":material/battery_charging_full:",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_db()
    _auth_sidebar()


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='pp-hero'><p class='pp-kicker'>POWERPULSE</p>"
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f"<div class='pp-note'>{text}</div>", unsafe_allow_html=True)


def current_user() -> dict | None:
    return st.session_state.get("user")


def _auth_sidebar() -> None:
    with st.sidebar:
        st.markdown("**Account**")
        user = current_user()
        if user:
            st.write(user["username"])
            if st.button("Log out", use_container_width=True):
                st.session_state.pop("user", None)
                st.rerun()
            return
        st.caption("Optional. Guest assessments stay on this computer and are not tied to an account.")
        mode = st.radio("Account action", ["Log in", "Create account"], label_visibility="collapsed")
        username = st.text_input("Username", key="auth_username")
        password = st.text_input("Password", type="password", key="auth_password")
        if st.button(mode, use_container_width=True):
            try:
                if mode == "Create account":
                    create_user(username, password)
                    st.success("Account created. Log in to use it.")
                else:
                    authenticated = authenticate(username, password)
                    if authenticated is None:
                        st.error("Username or password does not match.")
                    else:
                        st.session_state["user"] = authenticated
                        st.rerun()
            except ValueError as exc:
                st.error(str(exc))
