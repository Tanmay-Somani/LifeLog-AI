import os
import chromadb
import pandas as pd
import hdbscan
from sklearn.ensemble import IsolationForest
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bertopic import BERTopic # Import for Topic Modeling
import spacy # Import for Named Entity Recognition

# --- Configuration ---
MIN_CLUSTER_SIZE = 2
CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"
OUTPUT_PLOT_FILENAME = "activity_clusters_nlp_enhanced.png"

# --- Main Analysis Function ---
def main():
    print("--- LifeLog-AI: NLP-Enhanced ML Analyzer ---")

    # --- 1. Load and Clean Data ---
    print(f"\n[Step 1/5] Loading and cleaning data from ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        data = collection.get(include=["metadatas", "embeddings", "documents"])
        if not data['ids']:
            print("[ERROR] Database is empty.")
            return

        df = pd.DataFrame(data['metadatas'])
        embeddings = np.array(data['embeddings'])
        documents = data['documents']

        initial_count = len(df)
        df = df[df['window'].str.strip() != '']
        df = df[~df['window'].str.startswith('Click:')]
        
        valid_indices = df.index.tolist()
        df = df.reset_index(drop=True)
        embeddings = embeddings[valid_indices]
        documents = [documents[i] for i in valid_indices] # Filter documents as well

        print(f"Successfully loaded and cleaned data. Using {len(df)} entries for analysis.")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # --- 2. Perform HDBSCAN Clustering (Structural Analysis) ---
    print(f"\n[Step 2/5] Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(embeddings)
    df['cluster'] = cluster_labels
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Discovered {num_clusters} structural clusters.")

    # --- 3. Perform Anomaly Detection (Unchanged) ---
    print(f"\n[Step 3/5] Performing Anomaly Detection...")
    iso_forest = IsolationForest(contamination='auto', random_state=42)
    anomaly_labels = iso_forest.fit_predict(embeddings)
    df['is_anomaly'] = (anomaly_labels == -1)
    print("Anomaly detection complete.")

    # --- 4. NEW: Perform NLP Analysis (Topic Modeling & NER) ---
    print(f"\n[Step 4/5] Performing NLP Analysis...")
    
    # --- A. Topic Modeling with BERTopic ---
    print("  - Discovering topics with BERTopic (this may take a few moments)...")
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size=2)
    topics, _ = topic_model.fit_transform(documents)
    df['topic'] = topics
    topic_info = topic_model.get_topic_info()
    print("  - Topic modeling complete.")
    
    # --- B. Named Entity Recognition with spaCy ---
    print("  - Extracting named entities with spaCy...")
    nlp = spacy.load("en_core_web_sm")
    # Combine relevant text fields for a comprehensive NER analysis
    df['full_text'] = df['text_summary'] + " " + df['vision_summary'] + " " + df.get('ocr_text', '')
    
    entities = []
    for text in nlp.pipe(df['full_text'], disable=["tok2vec", "parser", "attribute_ruler", "lemmatizer"]):
        # Extract PERSON, ORG (Organizations), and GPE (Geo-Political Entities)
        ents = [f"{ent.text} ({ent.label_})" for ent in ent.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
        entities.append(", ".join(ents) if ents else "None")
    df['entities'] = entities
    print("  - Named entity recognition complete.")

    # --- 5. Report and Visualize ---
    print("\n[Step 5/5] Generating reports and visualization...")
    
    # --- NLP Report: Topics ---
    print("\n" + "="*60)
    print("               NLP Topic Modeling Results (BERTopic)")
    print("="*60)
    print(topic_info[["Topic", "Count", "Name"]].to_string())
    
    # --- NLP Report: Entities ---
    print("\n" + "="*60)
    print("               NLP Named Entity Recognition Results (spaCy)")
    print("="*60)
    # Filter for rows where entities were actually found
    entity_df = df[df['entities'] != 'None']
    if not entity_df.empty:
        print("Found the following entities in your activities:")
        for index, row in entity_df.iterrows():
            print(f"  - In Window '{row['window']}': {row['entities']}")
    else:
        print("No significant named entities (Person, Org, Location) found.")
    
    # (Structural Cluster and Anomaly reports can be added back here if desired)
    # ...

    # --- Generate Visualization ---
    try:
        print(f"\nGenerating a 2D visualization...")
        reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        df_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        # We will now color the plot by the discovered NLP TOPIC for more meaning
        df_2d['topic'] = [topic_info[topic_info['Topic'] == t]['Name'].iloc[0] for t in topics]
        
        plt.figure(figsize=(16, 12))
        sns.scatterplot(
            data=df_2d, x='x', y='y', hue='topic', palette='turbo', s=70, alpha=0.8
        )
        plt.title('2D Visualization of Your Activity by NLP Topic (UMAP + BERTopic)', fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(title='Discovered Topic', bbox_to_anchor=(1.05, 1), loc=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        plt.savefig(OUTPUT_PLOT_FILENAME)
        print(f"Success! Visualization saved to '{OUTPUT_PLOT_FILENAME}'")
    except Exception as e:
        print(f"\n[ERROR] Failed to generate visualization: {e}")

if __name__ == "__main__":
    main()