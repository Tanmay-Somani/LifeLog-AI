import sqlite3
import os

DB_PATH = os.path.join("data", "user_interactions.db")

def check_database_logs():
    print("--- SQLite Database Log Inspector ---")
    
    if not os.path.exists(DB_PATH):
        print("[ERROR] Database file not found. Please run logger.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Fetch the most recent 20 entries to see what's being logged
        cursor.execute("SELECT id, timestamp_utc, event_type, screenshot_path FROM interactions ORDER BY id DESC LIMIT 20")
        rows = cursor.fetchall()
        
        if not rows:
            print("[INFO] The database is empty.")
            return

        print("\nDisplaying the 20 most recent log entries:")
        print("-" * 60)
        print(f"{'ID':<5} | {'Timestamp (UTC)':<20} | {'Event Type':<12} | Screenshot Path")
        print("-" * 60)

        found_a_path = False
        for row in rows:
            log_id, ts, event, path = row
            if path:
                found_a_path = True
            print(f"{log_id:<5} | {ts:<20} | {event:<12} | {path}")

        print("-" * 60)
        
        # Final analysis
        print("\n--- Analysis ---")
        if not found_a_path:
            print("[CRITICAL FAILURE] No screenshot paths were found in the recent logs.")
            print("This confirms the logger is creating images but failing to save the file path to the database.")
            print("This is the reason the batch processor is not analyzing any images.")
        else:
            print("[SUCCESS] Screenshot paths WERE found in the database.")
            print("This would indicate a more complex issue, but is unlikely based on your logs.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred while reading the database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database_logs()
