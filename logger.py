import sqlite3
import time
import os
import threading
from pynput import mouse, keyboard
import mss
from PIL import Image
from win32gui import GetForegroundWindow, GetWindowText

# --- Configuration ---
DB_PATH = os.path.join("data", "user_interactions.db")
IMAGE_DIR = "images"
SCREENSHOT_MAX_WIDTH = 960
SCREENSHOT_JPEG_QUALITY = 60
# How often the main loop processes the event buffer
BATCH_INTERVAL_SECONDS = 2.0 
SCREENSHOT_COOLDOWN_SECONDS = 30

# --- A thread-safe buffer for raw events ---
event_buffer = []
buffer_lock = threading.Lock()

# --- Setup and Capture Functions ---
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, timestamp_utc TEXT, event_type TEXT NOT NULL,
            event_details TEXT, active_window TEXT, screenshot_path TEXT,
            processed INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def capture_and_save_optimized():
    try:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(IMAGE_DIR, f"{timestamp_str}_{int(time.time() * 1000)}.jpg")
        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img = img.convert("L")
            if img.width > SCREENSHOT_MAX_WIDTH:
                aspect_ratio = img.height / img.width
                new_height = int(SCREENSHOT_MAX_WIDTH * aspect_ratio)
                img = img.resize((SCREENSHOT_MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=SCREENSHOT_JPEG_QUALITY)
        return output_path
    except Exception as e:
        print(f"[ERROR] Could not capture screenshot: {e}")
        return None

# --- Listeners (Producers) ---
# These are now extremely simple: just add raw data to the buffer.
def on_click(x, y, button, pressed):
    if pressed:
        with buffer_lock:
            details = f"Click: {str(button)} at ({x}, {y})"
            event_buffer.append((time.time(), "mouse_click", details))

def on_press(key):
    with buffer_lock:
        try:
            details = f"Key Press: '{key.char}'"
        except AttributeError:
            details = f"Special Key: {str(key)}"
        event_buffer.append((time.time(), "keystroke", details))

# --- Main Application Logic ---
def main():
    print("--- User Interaction Logger (Batch Processor Version) ---")
    if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)
    setup_database()

    mouse_listener = mouse.Listener(on_click=on_click, on_press=on_press)
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener.start()
    keyboard_listener.start()

    print("[INFO] Logger is running. Press Ctrl+C to stop.")
    
    last_active_window = GetWindowText(GetForegroundWindow())
    last_screenshot_time = 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        while True:
            time.sleep(BATCH_INTERVAL_SECONDS)

            with buffer_lock:
                if not event_buffer:
                    continue
                # Safely copy and clear the buffer
                events_to_process = event_buffer[:]
                event_buffer.clear()

            current_window_title = GetWindowText(GetForegroundWindow())
            path_for_db = None
            
            # --- Screenshot logic now applies to the entire batch ---
            window_has_changed = current_window_title != last_active_window
            cooldown_has_passed = (time.time() - last_screenshot_time) > SCREENSHOT_COOLDOWN_SECONDS
            # Check if any mouse clicks are in the batch
            mouse_click_in_batch = any(evt[1] == 'mouse_click' for evt in events_to_process)

            if window_has_changed:
                print(f"\n[INFO] Window changed to '{current_window_title}'. Triggering screenshot for batch.")
                path_for_db = capture_and_save_optimized()
                last_screenshot_time = time.time()
                last_active_window = current_window_title
            elif mouse_click_in_batch and cooldown_has_passed:
                print(f"\n[INFO] Click detected in batch after cooldown. Triggering screenshot for batch.")
                path_for_db = capture_and_save_optimized()
                last_screenshot_time = time.time()

            # --- Write the entire batch to the database ---
            records_to_insert = []
            for ts, event_type, details in events_to_process:
                utc_ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))
                # The same screenshot path is applied to all events in this batch
                records_to_insert.append((ts, utc_ts, event_type, details, current_window_title, path_for_db))

            cursor.executemany('''
                INSERT INTO interactions (timestamp, timestamp_utc, event_type, event_details, active_window, screenshot_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', records_to_insert)
            conn.commit()
            print(f"> Logged batch of {len(records_to_insert)} events.", end='\r')

    except KeyboardInterrupt:
        print("\n[STOP] Stopping Logger...")
    finally:
        mouse_listener.stop()
        keyboard_listener.stop()
        conn.close()
        print("[INFO] Logger stopped successfully.")

if __name__ == "__main__":
    main()