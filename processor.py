import os
import sqlite3
import chromadb
import pandas as pd
import hdbscan
from sklearn.ensemble import IsolationForest
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bertopic import BERTopic
import spacy
from fpdf import FPDF, XPos, YPos
import logging
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re
import base64
import io
import pytesseract
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image
import imagehash
import colorama 

DB_PATH = os.path.join("data", "user_interactions.db")
CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"
OUTPUTS_DIR = "outputs"
FONTS_DIR = "fonts"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VISION_MODEL = "llava" 
CHUNK_TIMEOUT_SECONDS = 5
PERCEPTUAL_HASH_SIZE = 8
VISION_MODEL_TIMEOUT = 180 
TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MIN_CLUSTER_SIZE = 2
OUTPUT_PDF_FILENAME = os.path.join(OUTPUTS_DIR, f"LifeLog_AI_Analysis_{TIMESTAMP}.pdf")
OUTPUT_PLOT_FILENAME = os.path.join(OUTPUTS_DIR, f"activity_clusters_{TIMESTAMP}.png")
CLUSTER_LABELS = {
    0: "Coding & Development", 1: "Web Browsing", 2: "Design Work",
    3: "Command Line", 4: "Documentation", 5: "Miscellaneous"
}

class ColoredFormatter(logging.Formatter):
    """A custom logging formatter to add colors based on log level."""
    
    # Define color codes using colorama
    COLORS = {
        'WARNING': colorama.Fore.YELLOW,
        'INFO': colorama.Fore.GREEN,
        'DEBUG': colorama.Fore.CYAN,
        'CRITICAL': colorama.Fore.RED,
        'ERROR': colorama.Fore.RED
    }

    def format(self, record):
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, '') + log_message

# --- NEW: Setup Centralized Colored Logging ---
# Initialize colorama to work on Windows
colorama.init(autoreset=True)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a handler to print to console
handler = logging.StreamHandler()

# Create an instance of our custom formatter and set it on the handler
formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Remove any existing handlers and add our new colored one
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)

# ==============================================================================
# SECTION 1: BATCH PROCESSOR LOGIC
# ==============================================================================
def process_new_interactions(client, collection, embedding_function, vision_llm):
    logger.info("--- [PIPELINE STEP 1/2] Starting Batch Processing of New Interactions ---")
    # ... (rest of the function is the same, but uses 'logger' instead of 'logging')
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions WHERE processed = 0 ORDER BY timestamp ASC")
    interactions = cursor.fetchall()
    conn.close()

    if not interactions:
        logger.info("No new interactions to process.")
        return

    chunks = group_interactions_into_chunks(interactions)
    total_chunks = len(chunks)
    logger.info(f"Grouped {len(interactions)} new interactions into {total_chunks} chunks.")
    
    processed_interaction_ids = []
    processed_hashes = set()
    last_vision_summary = "No visual context yet."
    last_ocr_text = "No text read from screen yet."

    for i, chunk in enumerate(chunks):
        chunk_ids = [interaction[0] for interaction in chunk]
        last_interaction = chunk[-1]
        logger.info(f"--- Processing New Chunk {i+1}/{total_chunks} (ID: {chunk_ids[-1]}) ---")
        
        text_summary = summarize_chunk_text(chunk)
        vision_summary, ocr_text = "No screenshot.", "No screenshot."
        screenshot_path = next((interaction[6] for interaction in reversed(chunk) if interaction[6] and os.path.exists(interaction[6])), None)
        
        if screenshot_path:
            p_hash = calculate_perceptual_hash(screenshot_path)
            if p_hash and p_hash in processed_hashes:
                vision_summary, ocr_text = last_vision_summary, last_ocr_text
                logger.info("  [Analysis]: SKIPPED (Duplicate image hash)")
            elif p_hash:
                processed_hashes.add(p_hash)
                base64_image = image_to_base64(screenshot_path)
                if base64_image:
                    try:
                        prompt = "Describe this screenshot of a user's desktop concisely."
                        msg = vision_llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}])])
                        vision_summary = msg.content.strip()
                        last_vision_summary = vision_summary
                    except Exception as e:
                        logger.error(f"  [Vision]: FAILED. Reason: {e}")
                        vision_summary = "Vision analysis failed."
                
                ocr_text = perform_ocr_on_image(screenshot_path)
                last_ocr_text = ocr_text

        combined_text = (f"type: user_activity_log\nwindow: {last_interaction[5]}\n"
                         f"summary: {text_summary}\nvisual_context: {vision_summary}\nscreen_text: {ocr_text}")
        
        embedding = embedding_function.embed_query(combined_text)
        start_utc, end_utc = chunk[0][2], last_interaction[2]
        
        collection.add(
            ids=[str(last_interaction[0])], embeddings=[embedding], documents=[combined_text],
            metadatas=[{"start_timestamp": float(chunk[0][1]), "end_timestamp": float(last_interaction[1]),
                        "start_utc": start_utc, "end_utc": end_utc, "window": last_interaction[5], 
                        "text_summary": text_summary, "vision_summary": vision_summary, 
                        "ocr_text": ocr_text, "screenshot_path": screenshot_path or "N/A"}]
        )
        processed_interaction_ids.extend(chunk_ids)

    mark_as_processed(processed_interaction_ids)
    logger.info("--- Batch Processing Step Complete ---")

# ... (All other functions from the processor remain the same, just ensure they use logger.info, logger.error etc.)
def group_interactions_into_chunks(interactions):
    if not interactions: return []
    chunks, current_chunk = [], [interactions[0]]
    for i in range(1, len(interactions)):
        prev_time, current_time = float(interactions[i-1][1]), float(interactions[i][1])
        is_window_change = interactions[i][3] == 'window_change'
        if (current_time - prev_time > CHUNK_TIMEOUT_SECONDS) or is_window_change:
            chunks.append(current_chunk)
            current_chunk = [interactions[i]]
        else:
            current_chunk.append(interactions[i])
    chunks.append(current_chunk)
    return chunks
def summarize_chunk_text(chunk):
    active_window = chunk[-1][5]
    keystrokes = []
    for interaction in chunk:
        event_type, details = interaction[3], interaction[4]
        if event_type == 'keystroke':
            if "Key Press: '" in details:
                keystrokes.append(details.split("'")[1])
            elif "Special Key: " in details:
                key = details.split("Key.")[-1]
                if key == 'space': keystrokes.append(' ')
                else: keystrokes.append(f'[{key}]')
    raw_text = "".join(keystrokes)
    typed_text = re.sub(r'(\[.*?\]){2,}', '[Multiple Actions]', raw_text).replace('[backspace]','').replace('[enter]','\n').strip()
    return f"In window '{active_window}', user typed: '{typed_text}'" if typed_text else f"User activity in '{active_window}'."
def image_to_base64(path):
    try:
        with Image.open(path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception: return None
def calculate_perceptual_hash(path):
    try:
        with Image.open(path) as img: return imagehash.phash(img, hash_size=PERCEPTUAL_HASH_SIZE)
    except Exception: return None
def perform_ocr_on_image(path):
    try:
        text = pytesseract.image_to_string(Image.open(path), timeout=30)
        return re.sub(r'\n{2,}', '\n', text).strip()
    except Exception as e:
        logger.error(f"  [OCR]: FAILED. Reason: {e}")
        return "OCR failed."
def mark_as_processed(ids):
    if not ids: return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("UPDATE interactions SET processed = 1 WHERE id = ?", [(id,) for id in ids])
    conn.commit()
    conn.close()
    logger.info(f"Successfully marked {len(ids)} interactions as processed.")
class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font("DejaVu", "", os.path.join(FONTS_DIR, "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", os.path.join(FONTS_DIR, "DejaVuSans-Bold.ttf"))
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'LifeLog-AI Analysis Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 10, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)
    def chapter_body(self, data):
        self.set_font('DejaVu', '', 10)
        self.multi_cell(0, 5, data, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln()
    def add_dataframe(self, df):
        self.set_font('DejaVu', 'B', 8)
        header = list(df.columns)
        col_widths = [15, 15, 160]
        for i, h in enumerate(header):
            self.cell(col_widths[i], 7, str(h), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True)
        self.ln()
        self.set_font('DejaVu', '', 7)
        for _, row in df.iterrows():
            row_items = [str(item)[:100] for item in row]
            for i, item in enumerate(row_items):
                self.cell(col_widths[i], 6, item, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.ln()
        self.ln()
def run_analysis(client, collection):
    logger.info("--- [PIPELINE STEP 2/2] Starting Full Analysis of All Data ---")
    logger.info("Loading and cleaning all data from ChromaDB...")
    data = collection.get(include=["metadatas", "embeddings", "documents"])
    if not data['ids']: raise ValueError("ChromaDB is empty. Cannot perform analysis.")
    df = pd.DataFrame(data['metadatas'])
    embeddings = np.array(data['embeddings'])
    documents = data['documents']
    df = df[df['window'].str.strip() != '']
    df = df[~df['window'].str.startswith('Click:')]
    valid_indices = df.index.tolist()
    df = df.reset_index(drop=True)
    embeddings = embeddings[valid_indices]
    documents = [documents[i] for i in valid_indices]
    logger.info(f"Analyzing {len(df)} total entries.")
    logger.info("Performing NLP Topic Modeling and NER...")
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size=2, verbose=False)
    topics, _ = topic_model.fit_transform(documents)
    df['topic'] = topics
    topic_info = topic_model.get_topic_info()
    nlp = spacy.load("en_core_web_sm")
    df['full_text'] = (df['text_summary'].fillna('') + " " + df['vision_summary'].fillna('') + " " + df.get('ocr_text', pd.Series([''] * len(df))).fillna(''))
    entities = [", ".join([f"{ent.text} ({ent.label_})" for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]])
                for doc in nlp.pipe(df['full_text'], disable=["parser", "lemmatizer"])]
    df['entities'] = [ent if ent else "None" for ent in entities]
    logger.info("Performing NLP-Informed Clustering and Anomaly Detection...")
    topic_one_hot = pd.get_dummies(df['topic'], prefix='topic')
    combined_features = np.hstack((embeddings, topic_one_hot))
    df['cluster'] = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(combined_features)
    df['is_anomaly'] = (IsolationForest(random_state=42).fit_predict(combined_features) == -1)
    generate_pdf_report(df, topic_info)
    generate_visualization(embeddings, df['topic'], topic_info, df['is_anomaly'].values)
    train_and_save_classifier(df, embeddings)
    logger.info("--- Full Analysis Step Complete ---")
def generate_pdf_report(df, topic_info):
    logger.info("Generating PDF report...")
    pdf = PDFReport()
    pdf.set_fill_color(240, 240, 240)
    pdf.add_page()
    pdf.chapter_title("NLP Topic Modeling Results (BERTopic)")
    pdf.add_dataframe(topic_info[["Topic", "Count", "Name"]].head(15))
    pdf.chapter_title("NLP-Informed Cluster Analysis (HDBSCAN)")
    for label in sorted(df['cluster'].unique()):
        if label == -1: continue
        cluster_df = df[df['cluster'] == label]
        cluster_name = CLUSTER_LABELS.get(label, f"Unknown Cluster {label}")
        pdf.set_font('DejaVu', 'B', 10)
        pdf.cell(0, 10, f"Cluster {label}: {cluster_name} ({len(cluster_df)} items)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('DejaVu', '', 8)
        pdf.chapter_body("Most Common Applications:\n" + cluster_df['window'].value_counts().head(3).to_string())
        pdf.chapter_body("\nExample Activities:\n" + "\n".join([f"- {s}" for s in cluster_df['text_summary'].head(3)]))
        pdf.ln(5)
    pdf.add_page()
    pdf.chapter_title("Anomaly Detection Results")
    anomaly_df = df[df['is_anomaly']]
    if not anomaly_df.empty:
        for _, row in anomaly_df.iterrows():
            pdf.chapter_body(f"Window: {row['window']}\nSummary: {row['text_summary']}\n")
    else:
        pdf.chapter_body("No significant anomalies detected.")
    pdf.chapter_title("Named Entity Recognition Results (spaCy)")
    entity_df = df[df['entities'] != 'None'][['window', 'entities']]
    if not entity_df.empty:
        for _, row in entity_df.iterrows():
             pdf.chapter_body(f"In Window '{row['window']}': {row['entities']}")
    else:
         pdf.chapter_body("No significant entities found.")
    pdf.output(OUTPUT_PDF_FILENAME)
    logger.info(f"PDF report saved to '{OUTPUT_PDF_FILENAME}'")
def generate_visualization(embeddings, topics, topic_info, anomaly_labels):
    logger.info("Generating 2D visualization...")
    n_neighbors = min(15, len(embeddings) - 1)
    if n_neighbors < 2:
        logger.warning("Not enough data to generate UMAP plot.")
        return
    reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42, n_jobs=1)
    embeddings_2d = reducer.fit_transform(embeddings)
    df_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_2d['topic'] = [topic_info.loc[topic_info['Topic'] == t, 'Name'].iloc[0] for t in topics]
    df_2d['anomaly'] = ['Anomaly' if label == -1 else 'Normal' for label in anomaly_labels]
    plt.figure(figsize=(16, 12))
    sns.scatterplot(data=df_2d, x='x', y='y', hue='topic', style='anomaly', size='anomaly',
                    sizes=(50, 200), palette='turbo', alpha=0.8)
    plt.title('2D Visualization by NLP Topic', fontsize=16)
    plt.legend(title='Discovered Topic', bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILENAME)
    logger.info(f"Visualization saved to '{OUTPUT_PLOT_FILENAME}'")
def train_and_save_classifier(df, embeddings):
    logger.info("--- Training a Deep Learning Classifier ---")
    trainable_df = df[df['cluster'] != -1].copy()
    if len(trainable_df) < 10:
        logger.warning("Not enough clustered data to train. Skipping.")
        return
    trainable_embeddings = embeddings[trainable_df.index]
    trainable_df['task_label'] = trainable_df['cluster'].map(CLUSTER_LABELS)
    trainable_df.dropna(subset=['task_label'], inplace=True)
    if len(trainable_df['task_label'].unique()) < 2:
        logger.warning("Need at least two different task labels to train. Skipping.")
        return
    X, y = trainable_embeddings, pd.Categorical(trainable_df['task_label']).codes
    label_mapping = dict(enumerate(pd.Categorical(trainable_df['task_label']).categories))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu',
                          solver='adam', random_state=42, early_stopping=True, n_iter_no_change=20)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    report = classification_report(y_test, predictions, target_names=target_names, zero_division=0)
    logger.info("Classification Report:\n" + report)
    model_bundle = {"model": model, "label_mapping": label_mapping}
    model_path = os.path.join(OUTPUTS_DIR, "task_classifier.joblib")
    joblib.dump(model_bundle, model_path)
    logger.info(f"Successfully saved model bundle to '{model_path}'")

# ==============================================================================
# SECTION 3: MAIN EXECUTION
# ==============================================================================
def main():
    logger.info("====== LifeLog-AI Processing & Analysis Pipeline START ======")
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
    if not os.path.exists(FONTS_DIR):
        logger.error(f"Fonts directory not found at '{FONTS_DIR}'. Please create it and add DejaVuSans.ttf and DejaVuSans-Bold.ttf.")
        return

    try:
        # Initialize models once
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vision_llm = ChatOllama(model=VISION_MODEL, temperature=0, request_timeout=VISION_MODEL_TIMEOUT)
        
        process_new_interactions(client, collection, embedding_function, vision_llm)
        run_analysis(client, collection)

    except Exception as e:
        logger.error(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)

    logger.info("====== LifeLog-AI Processing & Analysis Pipeline END ======")

if __name__ == "__main__":
    main()