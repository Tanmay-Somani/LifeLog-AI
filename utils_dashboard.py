import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkfont
import sqlite3
import chromadb
import os
import threading
import queue
import time
import joblib
from langchain_community.embeddings import SentenceTransformerEmbeddings
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# --- Universal Configuration ---
DB_PATH = os.path.join("data", "user_interactions.db")
CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"
MODEL_PATH = os.path.join("outputs", "task_classifier.joblib")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MONITOR_INTERVAL_SECONDS = 5

# --- Main Application Class ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LifeLog-AI Master Dashboard")
        self.geometry("1100x800")
        
        self.gui_queue = queue.Queue()
        self.monitor_thread = None
        self.monitor_stop_event = threading.Event()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.create_explorer_tab() # New Visual Explorer Tab
        self.create_live_monitor_tab()
        self.create_check_sqlite_tab()
        self.create_verify_chroma_tab()
        self.create_reset_sqlite_tab()
        
        self.after(100, self.process_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    # --- Core App Logic (Queue, Threading, Closing) ---
    def on_closing(self):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_stop_event.set()
        self.destroy()

    def process_queue(self):
        try:
            while True:
                task = self.gui_queue.get(block=False)
                task_name, args = task[0], task[1:]
                if hasattr(self, task_name):
                    getattr(self, task_name)(*args)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def update_text_widget(self, widget, text, clear_first=False):
        widget.config(state=tk.NORMAL)
        if clear_first: widget.delete('1.0', tk.END)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def start_worker_thread(self, target_function, *args):
        thread = threading.Thread(target=target_function, args=args, daemon=True)
        thread.start()
        return thread

    # --- NEW: Tab 1: Visual Explorer ---
    def create_explorer_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Visual Explorer")

        # Top frame for controls
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.search_query = tk.StringVar()
        search_entry = ttk.Entry(control_frame, textvariable=self.search_query, width=60)
        search_entry.pack(side='left', fill='x', expand=True)
        search_button = ttk.Button(control_frame, text="Search", command=self.run_explorer_search)
        search_button.pack(side='left', padx=5)
        refresh_button = ttk.Button(control_frame, text="Load Recent", command=self.run_explorer_recent)
        refresh_button.pack(side='left', padx=5)

        # Canvas for scrollable results
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def run_explorer_search(self):
        query = self.search_query.get()
        if query:
            self.start_worker_thread(self._execute_search, query)

    def run_explorer_recent(self):
        self.start_worker_thread(self._execute_load_recent)
        
    def _update_explorer_results(self, results, title):
        # This function is called from the GUI thread via the queue
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        ttk.Label(self.scrollable_frame, text=title, font=tkfont.Font(size=14, weight='bold')).pack(anchor='w', pady=10)
        
        if not results:
            ttk.Label(self.scrollable_frame, text="No results found.").pack()
            return

        for item in results:
            self.create_result_card(item)

    def create_result_card(self, item):
        card = ttk.Frame(self.scrollable_frame, padding=10, borderwidth=1, relief="solid")
        card.pack(fill='x', padx=10, pady=5)
        
        # Left side for text
        left_frame = ttk.Frame(card)
        left_frame.pack(side='left', fill='x', expand=True, padx=10)
        
        # Right side for image
        right_frame = ttk.Frame(card, width=250)
        right_frame.pack(side='right')
        right_frame.pack_propagate(False)

        # Content
        ttk.Label(left_frame, text=item.get('card_title', ''), font=tkfont.Font(weight='bold')).pack(anchor='w')
        ttk.Label(left_frame, text=f"Window: {item.get('window', 'N/A')}", wraplength=500).pack(anchor='w')
        ttk.Label(left_frame, text="Text Summary:", font=tkfont.Font(slant='italic')).pack(anchor='w', pady=(5,0))
        ttk.Label(left_frame, text=item.get('text_summary', 'N/A'), wraplength=500, foreground="blue").pack(anchor='w')
        
        screenshot_path = item.get('screenshot_path')
        if screenshot_path and screenshot_path != 'N/A' and os.path.exists(screenshot_path):
            try:
                img = Image.open(screenshot_path)
                img.thumbnail((250, 250))
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(right_frame, image=photo)
                img_label.image = photo # Keep a reference
                img_label.pack()
            except Exception as e:
                ttk.Label(right_frame, text=f"Could not load image: {e}").pack()
        else:
            ttk.Label(right_frame, text="No screenshot.").pack()

    def _execute_search(self, query):
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
            results_data = collection.query(query_texts=[query], n_results=10, include=["metadatas", "distances"])
            
            results_list = []
            for meta, dist in zip(results_data['metadatas'][0], results_data['distances'][0]):
                meta['card_title'] = f"Result (Score: {1 - dist:.2f})"
                results_list.append(meta)
            
            self.gui_queue.put(("_update_explorer_results", results_list, f"Search Results for: '{query}'"))
        except Exception as e:
            self.gui_queue.put(("_update_explorer_results", [], f"Search failed: {e}"))

    def _execute_load_recent(self):
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
            results_data = collection.get(include=["metadatas"])
            df = pd.DataFrame(results_data['metadatas'])
            df['end_timestamp'] = pd.to_numeric(df['end_timestamp'])
            df_sorted = df.sort_values(by="end_timestamp", ascending=False).head(10)
            
            results_list = df_sorted.to_dict('records')
            self.gui_queue.put(("_update_explorer_results", results_list, "10 Most Recent Activities"))
        except Exception as e:
            self.gui_queue.put(("_update_explorer_results", [], f"Failed to load recent data: {e}"))
            
    # --- Live Monitor Tab ---
    def create_live_monitor_tab(self):
        # ... (code for this tab is unchanged)
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Live Activity Monitor")
        self.monitor_output = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        self.monitor_output.pack(padx=10, pady=10, fill="both", expand=True)
        self.monitor_button = ttk.Button(tab, text="Start Monitor", command=self.toggle_monitor)
        self.monitor_button.pack(pady=5)

    def toggle_monitor(self):
        # ... (code for this function is unchanged)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.update_output(self.monitor_output, "\n--- Sending stop signal... ---\n")
            self.monitor_stop_event.set()
            self.monitor_button.config(text="Start Monitor")
        else:
            self.monitor_stop_event.clear()
            self.monitor_thread = self.start_worker_thread(self.run_live_monitor, self.monitor_output)
            self.monitor_button.config(text="Stop Monitor")

    def run_live_monitor(self, output_widget):
        # ... (code for this function is unchanged)
        self.update_output(output_widget, "--- Starting Live Activity Monitor ---\n", clear_first=True)
        try:
            model_bundle = joblib.load(MODEL_PATH)
            model, label_mapping = model_bundle["model"], model_bundle["label_mapping"]
            self.update_output(output_widget, f"[INFO] Loaded model and mapping: {label_mapping}\n")
            embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            self.update_output(output_widget, f"[ERROR] Failed to load model: {e}\n")
            self.monitor_button.config(text="Start Monitor")
            return
        last_processed_id = 0
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        while not self.monitor_stop_event.is_set():
            cursor.execute("SELECT event_type, event_details, active_window FROM interactions WHERE id > ? ORDER BY timestamp ASC", (last_processed_id,))
            new_interactions = cursor.fetchall()
            if new_interactions:
                cursor.execute("SELECT MAX(id) FROM interactions")
                last_processed_id = cursor.fetchone()[0]
                text_summary = self.summarize_live_text(new_interactions)
                embedding = embedding_function.embed_query(text_summary)
                pred_code = model.predict(np.array(embedding).reshape(1, -1))[0]
                pred_label = label_mapping.get(pred_code, "Unknown")
                confidence = np.max(model.predict_proba(np.array(embedding).reshape(1, -1))) * 100
                self.update_output(output_widget, f"\n--- {time.strftime('%H:%M:%S')} ---\nActivity: {text_summary}\n==> Predicted Task: {pred_label} (Confidence: {confidence:.2f}%)\n")
            time.sleep(MONITOR_INTERVAL_SECONDS)
        conn.close()
        self.update_output(output_widget, "\n--- Monitor has stopped. ---\n")
        
    def summarize_live_text(self, interactions):
        # ... (code for this function is unchanged)
        if not interactions: return ""
        active_window = interactions[-1][2]
        keystrokes = []
        for event_type, details, _ in interactions:
            if event_type == 'keystroke':
                if "Key Press: '" in details:
                    keystrokes.append(details.split("'")[1])
                elif "Special Key: " in details:
                    key = details.split("Key.")[-1]
                    if key == 'space': keystrokes.append(' ')
                    else: keystrokes.append(f'[{key}]')
        typed_text = "".join(keystrokes).replace('[backspace]','').strip()
        return f"In '{active_window}', typed: '{typed_text}'" if typed_text else f"Activity in '{active_window}'"

    # --- Other Utility Tabs (SQLite Check, Chroma Verify, Reset DB) ---
    def create_check_sqlite_tab(self):
        # ... (code is unchanged)
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Check SQLite Logs")
        output_area = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        output_area.pack(padx=10, pady=10, fill="both", expand=True)
        button = ttk.Button(tab, text="Check Last 20 SQLite Logs", command=lambda: self.start_worker_thread(self.run_check_sqlite, output_area))
        button.pack(pady=5)
    def run_check_sqlite(self, output_widget):
        # ... (code is unchanged)
        self.update_output(output_widget, "--- Checking SQLite Database ---\n\n", clear_first=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp_utc, event_type, screenshot_path FROM interactions ORDER BY id DESC LIMIT 20")
        rows = cursor.fetchall()
        conn.close()
        header = f"{'ID':<5} | {'Timestamp (UTC)':<20} | {'Event Type':<12} | Screenshot Path\n" + "-" * 70 + "\n"
        self.update_output(output_widget, header)
        for row in rows: self.update_output(output_widget, f"{row[0]:<5} | {row[1] or 'N/A':<20} | {row[2]:<12} | {row[3]}\n")

    def create_verify_chroma_tab(self):
        # ... (code is unchanged)
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Verify ChromaDB")
        output_area = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        output_area.pack(padx=10, pady=10, fill="both", expand=True)
        button = ttk.Button(tab, text="Run ChromaDB Verification", command=lambda: self.start_worker_thread(self.run_verify_chroma, output_area))
        button.pack(pady=5)
    def run_verify_chroma(self, output_widget):
        # ... (code is unchanged)
        self.update_output(output_widget, "--- Verifying ChromaDB ---\n\n", clear_first=True)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        item_count = collection.count()
        self.update_output(output_widget, f"[INFO] Database contains {item_count} items.\n\n")
        if item_count == 0: return
        results = collection.get(limit=5, include=["metadatas", "documents"])
        for i, doc_id in enumerate(results['ids']):
            self.update_output(output_widget, f"-- Item {i+1} (ID: {doc_id}) --\n[SUCCESS] Doc Content: {results['documents'][i][:200]}...\n  Metadata: {results['metadatas'][i]}\n")
            
    def create_reset_sqlite_tab(self):
        # ... (code is unchanged)
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reset SQLite DB")
        output_area = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        output_area.pack(padx=10, pady=10, fill="both", expand=True)
        button = ttk.Button(tab, text="Reset 'Processed' Flags", command=self.confirm_and_execute_reset_wrapper)
        button.pack(pady=5)
    def confirm_and_execute_reset_wrapper(self):
        output_widget = self.notebook.winfo_children()[4].winfo_children()[0]
        self.start_worker_thread(self.run_reset_sqlite, output_widget)
    def run_reset_sqlite(self, output_widget):
        self.after(0, lambda: self.confirm_and_execute_reset(output_widget))
    def confirm_and_execute_reset(self, output_widget):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all 'processed' flags to 0?"):
            self.start_worker_thread(self.execute_reset, output_widget)
        else:
            self.update_output(output_widget, "Reset cancelled.\n")
    def execute_reset(self, output_widget):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE interactions SET processed = 0")
        rows = cursor.rowcount
        conn.commit()
        conn.close()
        self.update_output(output_widget, f"[SUCCESS] Reset {rows} rows.\n")

if __name__ == "__main__":
    app = App()
    app.mainloop()