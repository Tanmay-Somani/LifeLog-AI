import sqlite3
import os
DB_PATH = os.path.join("data", "user_interactions.db")

if not os.path.exists(DB_PATH):
    print("Database not found. Nothing to reset.")
else:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE interactions SET processed = 0")
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Database reset successfully. Marked {updated_rows} rows as unprocessed.")