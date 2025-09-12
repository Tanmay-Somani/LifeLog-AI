import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sqlite3
import chromadb
import os
import threading
import queue

# --- Configuration (Centralized for the app) ---
DB_PATH = os.path.join("data", "user_interactions.db")
CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"

# --- Main Application Class ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("LifeLog-AI Utility Dashboard")
        self.geometry("800x600")

        # --- Create a Tabbed Interface ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.create_check_sqlite_tab()
        self.create_verify_chroma_tab()
        self.create_reset_sqlite_tab()
        
        # Queue for thread communication
        self.gui_queue = queue.Queue()
        self.after(100, self.process_queue)

    def process_queue(self):
        try:
            message = self.gui_queue.get(block=False)
            # Unpack the message from the queue
            widget, text, clear = message
            if clear:
                widget.config(state=tk.NORMAL)
                widget.delete('1.0', tk.END)
            widget.insert(tk.END, text)
            widget.see(tk.END) # Auto-scroll
            widget.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def update_output(self, widget, text, clear=False):
        # Put a message into the queue for the GUI thread to process
        self.gui_queue.put((widget, text, clear))

    # --- Tab 1: Check SQLite Logs ---
    def create_check_sqlite_tab(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Check SQLite Logs")

        output_area = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, state=tk.DISABLED)
        output_area.pack(padx=10, pady=10, fill="both", expand=True)

        button = ttk.Button(tab1, text="Check Last 20 SQLite Logs", 
                            command=lambda: self.start_thread(self.run_check_sqlite, output_area))
        button.pack(pady=5)

    def run_check_sqlite(self, output_widget):
        self.update_output(output_widget, "--- Checking SQLite Database ---\n\n", clear=True)
        if not os.path.exists(DB_PATH):
            self.update_output(output_widget, "[ERROR] Database file not found.")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, timestamp_utc, event_type, screenshot_path FROM interactions ORDER BY id DESC LIMIT 20")
            rows = cursor.fetchall()
            
            if not rows:
                self.update_output(output_widget, "[INFO] The database is empty.\n")
                return

            header = f"{'ID':<5} | {'Timestamp (UTC)':<20} | {'Event Type':<12} | Screenshot Path\n"
            separator = "-" * 60 + "\n"
            self.update_output(output_widget, header + separator)

            for row in rows:
                log_id, ts, event, path = row
                self.update_output(output_widget, f"{log_id:<5} | {ts or 'N/A':<20} | {event:<12} | {path}\n")
            self.update_output(output_widget, separator)
        except Exception as e:
            self.update_output(output_widget, f"[ERROR] An error occurred: {e}\n")
        finally:
            conn.close()

    # --- Tab 2: Verify ChromaDB ---
    def create_verify_chroma_tab(self):
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Verify ChromaDB")
        
        output_area = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, state=tk.DISABLED)
        output_area.pack(padx=10, pady=10, fill="both", expand=True)

        button = ttk.Button(tab2, text="Run ChromaDB Verification", 
                            command=lambda: self.start_thread(self.run_verify_chroma, output_area))
        button.pack(pady=5)

    def run_verify_chroma(self, output_widget):
        self.update_output(output_widget, "--- Verifying ChromaDB ---\n\n", clear=True)
        if not os.path.exists(CHROMA_PATH):
            self.update_output(output_widget, "[ERROR] ChromaDB path not found.\n")
            return
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
            item_count = collection.count()
            self.update_output(output_widget, f"[INFO] Database contains {item_count} items.\n\n")

            if item_count == 0: return

            self.update_output(output_widget, "Fetching the 5 most recent items to inspect...\n")
            results = collection.get(limit=5, include=["metadatas", "documents"])

            separator = "-" * 50 + "\n"
            for i, doc_id in enumerate(results['ids']):
                self.update_output(output_widget, separator)
                self.update_output(output_widget, f"--- Item {i+1} (ID: {doc_id}) ---\n")
                
                doc_content = results['documents'][i]
                if doc_content:
                    self.update_output(output_widget, "[SUCCESS] Document Content is PRESENT.\n")
                    self.update_output(output_widget, f"  Content: {doc_content[:200]}...\n")
                else:
                    self.update_output(output_widget, "[CRITICAL FAILURE] Document Content is MISSING or EMPTY.\n")
                
                self.update_output(output_widget, f"  Metadata: {results['metadatas'][i]}\n")
            self.update_output(output_widget, separator)
        except Exception as e:
            self.update_output(output_widget, f"[ERROR] An error occurred: {e}\n")

    # --- Tab 3: Reset SQLite ---
    def create_reset_sqlite_tab(self):
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Reset SQLite DB")

        output_area = scrolledtext.ScrolledText(tab3, wrap=tk.WORD, state=tk.DISABLED)
        output_area.pack(padx=10, pady=10, fill="both", expand=True)

        button = ttk.Button(tab3, text="Reset 'Processed' Flags", 
                            command=lambda: self.start_thread(self.run_reset_sqlite, output_area))
        button.pack(pady=5)
    
    def run_reset_sqlite(self, output_widget):
        self.update_output(output_widget, "--- Resetting SQLite 'processed' flags ---\n", clear=True)
        
        # Confirmation Dialog must be run in the main thread
        should_reset = messagebox.askyesno(
            "Confirm Reset", 
            "Are you sure you want to reset all 'processed' flags to 0?\n"
            "This will cause the batch_processor to re-process all interactions."
        )

        if not should_reset:
            self.update_output(output_widget, "Reset cancelled by user.\n")
            return

        if not os.path.exists(DB_PATH):
            self.update_output(output_widget, "[ERROR] Database not found.\n")
            return
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("UPDATE interactions SET processed = 0")
            updated_rows = cursor.rowcount
            conn.commit()
            conn.close()
            self.update_output(output_widget, f"[SUCCESS] Database reset. Marked {updated_rows} rows as unprocessed.\n")
        except Exception as e:
            self.update_output(output_widget, f"[ERROR] An error occurred: {e}\n")
    
    # --- Threading Helper ---
    def start_thread(self, target_func, *args):
        # Starts a new thread for any long-running task to keep the GUI responsive
        thread = threading.Thread(target=target_func, args=args, daemon=True)
        thread.start()

if __name__ == "__main__":
    app = App()
    app.mainloop()
