import sqlite3
import time
import os
import threading
import queue
from pynput import mouse, keyboard
import mss
from PIL import Image
from win32gui import GetForegroundWindow, GetWindowText

DB_PATH = os.path.join("data", "user_interactions.db")
IMAGE_DIR = "images"
SCREENSHOT_MAX_WIDTH = 960
SCREENSHOT_JPEG_QUALITY = 60
SCREENSHOT_COOLDOWN_SECONDS = 30

event_queue = queue.Queue()

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            timestamp_utc TEXT, 
            event_type TEXT NOT NULL,
            event_details TEXT,
            active_window TEXT,
            screenshot_path TEXT,
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
        # On success, return the path
        return output_path
    except Exception as e:
        print(f"[ERROR] Could not capture screenshot: {e}")
        # On failure, return None
        return None

def database_writer_thread():
    last_active_window = GetWindowText(GetForegroundWindow())
    last_screenshot_time = 0 
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    while True:
        event = event_queue.get()
        if event is None: break

        event_type, details = event
        
        unix_timestamp = time.time()
        utc_timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(unix_timestamp))
        current_window_title = GetWindowText(GetForegroundWindow())
        path_for_db = None # Start with None

        window_has_changed = current_window_title and current_window_title != last_active_window
        cooldown_has_passed = (unix_timestamp - last_screenshot_time) > SCREENSHOT_COOLDOWN_SECONDS

        should_take_screenshot = False
        if window_has_changed:
            print(f"\n[INFO] Window changed to '{current_window_title}'. Triggering screenshot.")
            last_active_window = current_window_title
            should_take_screenshot = True
        elif event_type == 'mouse_click' and cooldown_has_passed:
            print(f"\n[INFO] Click detected after cooldown. Triggering screenshot.")
            should_take_screenshot = True
            
        if should_take_screenshot:
            # ONLY call the function if we need to.
            # The function returns the path on success or None on failure.
            path_for_db = capture_and_save_optimized()
            last_screenshot_time = time.time()
        
        cursor.execute('''
            INSERT INTO interactions (timestamp, timestamp_utc, event_type, event_details, active_window, screenshot_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (unix_timestamp, utc_timestamp_str, event_type, str(details), current_window_title, path_for_db))
        conn.commit()
        
        print(f"> Logged: {event_type:11} | Window: {current_window_title[:90]}", end='\r')
    conn.close()
    print("\n[INFO] Database writer thread has shut down.")

def on_click(x, y, button, pressed):
    if pressed:
        details = f"Click: {str(button)} at ({x}, {y})"
        event_queue.put(("mouse_click", details))
def on_press(key):
    try:
        details = f"Key Press: '{key.char}'"
    except AttributeError:
        details = f"Special Key: {str(key)}"
    event_queue.put(("keystroke", details))
if __name__ == "__main__":
    print("--- User Interaction Logger ---")
    if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)
    setup_database()
    writer_thread = threading.Thread(target=database_writer_thread, daemon=True)
    writer_thread.start()
    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener.start()
    keyboard_listener.start()
    print(f"[INFO] Logger is running.")
    print(f"[INFO] Press Ctrl+C in this terminal to gracefully stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Stopping Logger...")
        mouse_listener.stop()
        keyboard_listener.stop()
        event_queue.put(None)
        writer_thread.join() 
        print("[INFO] Logger stopped successfully.")