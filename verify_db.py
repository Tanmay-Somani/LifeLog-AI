import chromadb
import os

CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"

def verify_database_contents():
    print("--- ChromaDB Verification Tool ---")
    
    if not os.path.exists(CHROMA_PATH):
        print("[ERROR] ChromaDB path not found. Did the batch_processor.py run?")
        return
    try:
        print(f"Connecting to database at: {CHROMA_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"[ERROR] Could not connect to the database: {e}")
        return

    item_count = collection.count()
    print(f"[INFO] Database contains {item_count} items.")

    if item_count == 0:
        print("[WARNING] The database is empty. This is likely the cause of the problem.")
        return

    print("\nFetching the 5 most recent items to inspect their contents...")
    
    try:

        results = collection.get(
            limit=5,
            include=["metadatas", "documents"] 
        )

        if not results['ids']:
            print("[ERROR] The database has items, but failed to retrieve any. The DB might be corrupt.")
            return

        print("-" * 50)
        for i, doc_id in enumerate(results['ids']):
            print(f"--- Item {i+1} (ID: {doc_id}) ---")

            document_content = results['documents'][i]
            if document_content:
                print(f"  [SUCCESS] Document Content is PRESENT.")
                print(f"  Content: {document_content[:200]}...") 
            else:
                print(f"  [CRITICAL FAILURE] Document Content is MISSING or EMPTY (None).")
                print(f"  This is the reason your searches are failing.")

            metadata_content = results['metadatas'][i]
            if metadata_content:
                print(f"  [INFO] Metadata is present.")
                print(f"  Metadata: {metadata_content}")
            else:
                 print(f"  [WARNING] Metadata is missing.")
            
            print("-" * 50)

    except Exception as e:
        print(f"\n[ERROR] An error occurred while fetching data: {e}")
        print("This could indicate a problem with the database structure.")

if __name__ == "__main__":
    verify_database_contents()
