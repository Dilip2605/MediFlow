import sqlite3
import hashlib
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# Purpose: Create all tables when project runs first time
# Run this file ONCE before starting the app
# Command: python database/init_db.py
# ═══════════════════════════════════════════════════════════════

def create_tables(conn):
    """Create all 5 tables needed by MediFlow"""
    cursor = conn.cursor()

    # ─── TABLE 1: PATIENTS ───────────────────────────────────
    # Stores basic patient identity information
    # Every other table links back to this via patient_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT    NOT NULL,
            age     INTEGER NOT NULL,
            gender  TEXT    NOT NULL,
            phone   TEXT    NOT NULL
        )
    """)
    
    cursor.execute(""" 
                   
     CREATE TABLE appointments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL,
    doctor_name TEXT    NOT NULL,
    department  TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    time_slot   TEXT    NOT NULL,
    status      TEXT    DEFAULT 'Scheduled',
    notes       TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)
                   """)

    # ─── TABLE 2: XRAY RESULTS ───────────────────────────────
    # Stores CNN X-ray analysis results
    # patient_id links to patients table (foreign key)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xray_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  INTEGER NOT NULL,
            disease     TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            date        TEXT    DEFAULT (date('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)

    # ─── TABLE 3: SYMPTOMS ───────────────────────────────────
    # Stores NLP symptom analysis records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symptoms (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER NOT NULL,
            symptoms_text   TEXT    NOT NULL,
            diagnosis       TEXT    NOT NULL,
            date            TEXT    DEFAULT (date('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)
    cursor.execute(""" 
     CREATE TABLE vitals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    temperature     REAL,
    blood_pressure  TEXT,
    pulse_rate      INTEGER,
    spo2            REAL,
    weight          REAL,
    recorded_by     TEXT,
    recorded_at     TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)       
            """)
    
    cursor.execute(""" 
                   
    CREATE TABLE lab_tests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    test_name       TEXT    NOT NULL,
    ordered_by      TEXT    NOT NULL,
    status          TEXT    DEFAULT 'Ordered',
    result_value    TEXT,
    normal_range    TEXT,
    result_date     TEXT,
    ordered_date    TEXT    DEFAULT (date('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)""")
    # ─── TABLE 4: INVENTORY ──────────────────────────────────
    # Stores hospital medicine stock
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name   TEXT    UNIQUE NOT NULL,
            quantity        INTEGER NOT NULL DEFAULT 0,
            last_updated    TEXT    DEFAULT (date('now'))
        )
    """)
    
    cursor.execute(""" 
    CREATE TABLE prescriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    doctor_name     TEXT    NOT NULL,
    diagnosis       TEXT    NOT NULL,
    medicines       TEXT    NOT NULL,
    dosage          TEXT    NOT NULL,
    duration_days   INTEGER NOT NULL,
    date            TEXT    DEFAULT (date('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)""")
    
    
    cursor.execute(""" 
     CREATE TABLE bills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    consultation    REAL    DEFAULT 500,
    xray_charge     REAL    DEFAULT 0,
    medicine_charge REAL    DEFAULT 0,
    lab_charge      REAL    DEFAULT 0,
    total_amount    REAL    NOT NULL,
    paid_status     TEXT    DEFAULT 'Pending',
    date            TEXT    DEFAULT (date('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(id)
      )
                   """)

    cursor.execute(""" 
     CREATE TABLE doctor_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL,
    doctor_name TEXT    NOT NULL,
    note_text   TEXT    NOT NULL,
    note_type   TEXT    DEFAULT 'General',
    date        TEXT    DEFAULT (date('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)""")
    # ─── TABLE 5: USERS ──────────────────────────────────────
    # Stores login credentials
    # Password stored as SHA256 hash — NEVER plain text!
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT    UNIQUE NOT NULL,
            password_hash   TEXT    NOT NULL,
            role            TEXT    DEFAULT 'doctor',
            created_at      TEXT    DEFAULT (date('now'))
        )
    """)

    conn.commit()
    print("✅ All tables created successfully!")


def create_default_admin(conn):
    """
    Create default admin user if no users exist.
    Why: Without this, nobody can login on first run!
    Default: username=admin, password=admin123
    CHANGE THIS PASSWORD after first login!
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]

    if count == 0:
        # Hash password with SHA256
        # SHA256 is one-way — cannot reverse to get plain text
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()

        cursor.execute("""
            INSERT INTO users (username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?)
        """, ("admin", password_hash, "admin", datetime.now().strftime("%Y-%m-%d")))

        conn.commit()
        print("✅ Default admin created!")
        print("   Username: admin")
        print("   Password: admin123")
        print("   ⚠️  Change password after first login!")
    else:
        print(f"ℹ️  {count} user(s) already exist — skipping default admin creation")


def insert_sample_inventory(conn):
    """Insert sample medicines for demonstration"""
    cursor = conn.cursor()

    # Check if inventory is empty
    cursor.execute("SELECT COUNT(*) FROM inventory")
    if cursor.fetchone()[0] == 0:
        medicines = [
            ("Paracetamol 500mg", 150),
            ("Amoxicillin 250mg", 80),
            ("Ibuprofen 400mg", 120),
            ("Azithromycin 500mg", 45),
            ("Doxycycline 100mg", 60),
            ("Metformin 500mg", 200),
            ("Atorvastatin 10mg", 90),
            ("Amlodipine 5mg", 35),
            ("Omeprazole 20mg", 110),
            ("Cetirizine 10mg", 75),
        ]

        today = datetime.now().strftime("%Y-%m-%d")
        for med, qty in medicines:
            cursor.execute("""
                INSERT OR IGNORE INTO inventory (medicine_name, quantity, last_updated)
                VALUES (?, ?, ?)
            """, (med, qty, today))

        conn.commit()
        print("✅ Sample inventory inserted!")
    else:
        print("ℹ️  Inventory already has data — skipping")


def main():
    """Main function — run all initialization steps"""
    print("=" * 50)
    print("MEDIFLOW — DATABASE INITIALIZATION")
    print("=" * 50)

    try:
        conn = sqlite3.connect("MediFlow.db")
        conn.execute("PRAGMA foreign_keys = ON")

        create_tables(conn)
        create_default_admin(conn)
        insert_sample_inventory(conn)

        print("\n🎉 Database initialized successfully!")
        print("   Now run: streamlit run app.py")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
