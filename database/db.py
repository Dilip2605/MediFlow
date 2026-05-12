import sqlite3
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# Purpose: Single cached connection for entire app
# Why cached: Without cache, new connection every page reload
#             = memory leak + slow performance
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def get_db():
    """
    Returns cached SQLite database connection.
    Called by every file that needs database access.
    
    Usage:
        from database.db import get_db
        conn = get_db()
        cursor = conn.cursor()
    """
    conn = sqlite3.connect("MediFlow.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key support
    return conn
