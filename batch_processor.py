import sqlite3
import os
import chromadb
import logging
import re
import base64
import io
import pytesseract
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image
import imagehash

TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

DB_PATH = os.path.join("data", "user_interactions.db")
CHROMA_PATH = os.path.join("data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VISION_MODEL = "llava" 
CHUNK_TIMEOUT_SECONDS = 5
PERCEPTUAL_HASH_SIZE = 8
VISION_MODEL_TIMEOUT = 180 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Initializing models and database connections...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="user_activity_collection")
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vision_llm = ChatOllama(model=VISION_MODEL, temperature=0, request_timeout=VISION_MODEL_TIMEOUT)
logging.info(f"Using vision model: {VISION_MODEL} with a {VISION_MODEL_TIMEOUT}s timeout.")
logging.info("Initialization complete.")

def get_unprocessed_interactions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions WHERE processed = 0 ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def group_interactions_into_chunks(interactions):
    if not interactions: return []
    chunks = []
    current_chunk = [interactions[0]]
    for i in range(1, len(interactions)):
        prev_time = float(interactions[i-1][1])
        current_time = float(interactions[i][1])
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
                char = details.split("'")[1]
                keystrokes.append(char)
            elif "Special Key: " in details:
                special_key = details.split("Key.")[-1]
                if special_key == 'space': keystrokes.append(' ')
                else: keystrokes.append(f'[{special_key}]')
    raw_typed_text = "".join(keystrokes)
    typed_text = re.sub(r'(\[.*?\]){2,}', '[Multiple Actions]', raw_typed_text) 
    typed_text = typed_text.replace('[backspace]', '').replace('[enter]', '\n').strip() 
    return f"In window '{active_window}', user typed: '{typed_text.strip()}'" if typed_text else f"User activity in '{active_window}'."

def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception: return None

def calculate_perceptual_hash(image_path):
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img, hash_size=PERCEPTUAL_HASH_SIZE)
    except Exception: return None

def perform_ocr_on_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), timeout=30)
        cleaned_text = re.sub(r'\n{2,}', '\n', text).strip()
        return cleaned_text
    except Exception as e:
        logging.error(f"  [OCR]: FAILED for image {image_path}. Reason: {e}")
        return "OCR failed or timed out."

def mark_as_processed(interaction_ids):
    if not interaction_ids: return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("UPDATE interactions SET processed = 1 WHERE id = ?", [(id,) for id in interaction_ids])
    conn.commit()
    conn.close()
    logging.info(f"Successfully marked {len(interaction_ids)} interactions as processed.")

def main():
    logging.info("Starting batch processing job with sequential OCR.")
    interactions = get_unprocessed_interactions()
    if not interactions:
        logging.info("No new interactions to process. Exiting.")
        return
    
    chunks = group_interactions_into_chunks(interactions)
    total_chunks = len(chunks)
    logging.info(f"Grouped {len(interactions)} interactions into {total_chunks} chunks.")
    
    processed_interaction_ids = []
    processed_hashes_this_run = set()
    last_vision_summary = "No visual context yet."
    last_ocr_text = "No text read from screen yet."

    try:
        for i, chunk in enumerate(chunks):
            chunk_ids = [interaction[0] for interaction in chunk]
            last_interaction = chunk[-1]
            logging.info(f"--- Processing Chunk {i+1}/{total_chunks} (Ending with ID: {chunk_ids[-1]}) ---")
            
            text_summary = summarize_chunk_text(chunk)
            vision_summary = "No screenshot for this chunk."
            ocr_text = "No screenshot for this chunk."
            screenshot_path = next((interaction[6] for interaction in reversed(chunk) if interaction[6] and os.path.exists(interaction[6])), None)
            
            if screenshot_path:
                p_hash = calculate_perceptual_hash(screenshot_path)
                if p_hash and p_hash in processed_hashes_this_run:
                    vision_summary = last_vision_summary
                    ocr_text = last_ocr_text
                    logging.info("  [Analysis]: SKIPPED (Duplicate image hash)")
                elif p_hash:
                    processed_hashes_this_run.add(p_hash)
                    
                    base64_image = image_to_base64(screenshot_path)
                    if base64_image:
                        try:
                            logging.info("  [Vision]: Contacting vision model...")
                            prompt = "Describe this screenshot of a user's desktop concisely."
                            msg = vision_llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}])])
                            vision_summary = msg.content.strip()
                            last_vision_summary = vision_summary
                            logging.info(f"  [Vision]: Summary received.")
                        except Exception as e:
                            logging.error(f"  [Vision]: FAILED. Reason: {e}")
                            vision_summary = "Vision analysis failed."
                    
                    logging.info("  [OCR]: Reading text from screenshot...")
                    ocr_text = perform_ocr_on_image(screenshot_path)
                    last_ocr_text = ocr_text
                    logging.info("  [OCR]: Text reading complete.")

            combined_text = f"""type: user_activity_log
window: {last_interaction[5]}
summary: {text_summary}
visual_context: {vision_summary}
screen_text: {ocr_text}
"""
            embedding = embedding_function.embed_query(combined_text)
            
            start_utc = chunk[0][2]
            end_utc = last_interaction[2]
            
            collection.add(
                ids=[str(last_interaction[0])],
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[{"start_timestamp": float(chunk[0][1]), "end_timestamp": float(last_interaction[1]),
                            "start_utc": start_utc, "end_utc": end_utc,
                            "window": last_interaction[5], "text_summary": text_summary,
                            "vision_summary": vision_summary, "ocr_text": ocr_text,
                            "screenshot_path": screenshot_path or "N/A"}]
            )
            processed_interaction_ids.extend(chunk_ids)
            
    except KeyboardInterrupt:
        logging.warning("\nInterruption detected! Saving progress before exiting.")
    finally:
        logging.info("Executing final save operation...")
        mark_as_processed(processed_interaction_ids)
        logging.info("Batch processing job finished.")

if __name__ == "__main__":
    main()