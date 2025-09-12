import streamlit as st
import chromadb
import pandas as pd
import os
from PIL import Image

CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"

st.set_page_config(
    page_title="LifeLog-AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_chroma_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"Failed to load ChromaDB collection: {e}")
        st.warning("Please ensure the 'batch_processor.py' script has been run at least once.")
        return None

@st.cache_data(ttl=60)
def get_all_data(_collection):
    results = _collection.get(include=["metadatas"])
    if not results or not results['metadatas']:
        return pd.DataFrame()
    return pd.DataFrame(results['metadatas'])

def display_result_card(item, card_title=""):
    with st.container(border=True):
        if card_title:
            st.markdown(card_title)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Window:** `{item.get('window', 'N/A')}`")
            st.markdown("**Text Summary:**")
            st.info(item.get('text_summary', 'Not available.'))
            st.markdown("**Vision Summary:**")
            st.success(item.get('vision_summary', 'Not available.'))
            with st.expander("Show Full Metadata"):
                st.json(item, expanded=False)

        with col2:
            screenshot_path = item.get('screenshot_path')
            if screenshot_path and screenshot_path != 'N/A' and os.path.exists(screenshot_path):
                try:
                    image = Image.open(screenshot_path)
                    timestamp_str = item.get('end_utc') or pd.to_datetime(item.get('end_timestamp', 0), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                    st.image(image, caption=f"Screenshot at {timestamp_str}", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            else:
                st.caption("No screenshot available.")

def main():
    st.title("LifeLog-AI Visual Explorer")
    st.markdown("An interface to visually explore and search your activity history.")

    collection = load_chroma_collection()
    if collection is None:
        return

    total_items = collection.count()
    st.sidebar.title("Dashboard Controls")
    st.sidebar.success(f"**{total_items}** activity chunks indexed.")

    app_mode = st.sidebar.selectbox("Choose a View", ["Recent Activity", "Search & Explore"])

    if app_mode == "Recent Activity":
        st.header("Most Recent Activity")
        df = get_all_data(collection)
        if df.empty:
            st.warning("No data found in the database.")
            return

        sort_column = 'end_timestamp' if 'end_timestamp' in df.columns else 'timestamp'
        if sort_column in df.columns:
            df[sort_column] = pd.to_numeric(df[sort_column])
            df_sorted = df.sort_values(by=sort_column, ascending=False)
        else:
            st.warning("Could not find a timestamp column to sort by.")
            df_sorted = df

        num_to_fetch = st.slider("Number of recent items to display", 5, min(100, len(df_sorted)), 10)
        
        for index, row in df_sorted.head(num_to_fetch).iterrows():
            display_result_card(row.to_dict())

    elif app_mode == "Search & Explore":
        st.header("Search Your Activity History")
        query_text = st.text_input("Enter your semantic search query:", placeholder="e.g., working on the logger script in vscode")
        
        if query_text:
            num_results = st.slider("Number of search results", 1, 20, 5)
            try:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=num_results,
                    include=["metadatas", "distances"]
                )
                
                if not results['ids'][0]:
                    st.warning("No results found for your query.")
                    return

                st.subheader("Search Results")
                for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                    card_title = f"**Result {i+1}** (Similarity Score: {1 - distance:.2f})"
                    display_result_card(metadata, card_title)

            except Exception as e:
                st.error(f"An error occurred during search: {e}")

if __name__ == "__main__":
    main()