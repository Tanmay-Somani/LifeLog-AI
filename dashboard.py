import streamlit as st
import chromadb
import pandas as pd
import os
from PIL import Image

# --- Configuration ---
CHROMA_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "user_activity_collection"

# --- Page Configuration ---
st.set_page_config(
    page_title="User Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_chroma_collection():
    """Loads the ChromaDB collection. Cached to prevent reloading on every interaction."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"Failed to load ChromaDB collection: {e}")
        st.error("Please ensure the 'batch_processor.py' script has been run at least once.")
        return None

def main():
    st.title("User Activity Visual Explorer")
    st.markdown("A visual interface to explore the data collected by your User Analysis Ecosystem.")

    collection = load_chroma_collection()
    if collection is None:
        return

    total_items = collection.count()
    st.sidebar.success(f"**{total_items}** items in the database.")

    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose a view:", ["Recent Activity", "Search & Explore"])

    if app_mode == "Recent Activity":
        st.header("Most Recent Activity")
        st.markdown("A reverse-chronological view of your processed activity.")
        num_to_fetch = st.slider("Number of recent items to display:", 5, 100, 10)
        
        try:
            results = collection.get(include=["metadatas"])
            if not results['metadatas']:
                st.warning("No data found in the database.")
                return

            df = pd.DataFrame(results['metadatas'])
            
            # --- THIS BLOCK IS THE FIX ---
            # It intelligently finds the correct timestamp column to sort by.
            sort_column = None
            if 'end_timestamp' in df.columns:
                sort_column = 'end_timestamp'
                df[sort_column] = pd.to_numeric(df[sort_column])
            elif 'timestamp' in df.columns: # Fallback for older data format
                sort_column = 'timestamp'
                df[sort_column] = pd.to_numeric(df[sort_column])
            
            if sort_column:
                df_sorted = df.sort_values(by=sort_column, ascending=False).head(num_to_fetch)
            else:
                st.warning("Could not find a timestamp column to sort by. Displaying in default order.")
                df_sorted = df.head(num_to_fetch) # Show unsorted if no timestamp found
            # --- END OF FIX ---

            for index, row in df_sorted.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Use .get() for safety, providing a default value if a key is missing
                        st.markdown(f"**Window:** `{row.get('window', 'N/A')}`")
                        st.markdown(f"**Text Summary:**")
                        st.info(row.get('text_summary', 'Not available.'))
                        st.markdown(f"**Vision Summary:**")
                        st.success(row.get('vision_summary', 'Not available.'))
                        
                        with st.expander("Show Full Metadata"):
                            st.json(row.to_dict())

                    with col2:
                        screenshot_path = row.get('screenshot_path')
                        if screenshot_path and screenshot_path != 'N/A' and os.path.exists(screenshot_path):
                            try:
                                image = Image.open(screenshot_path)
                                # Use the sort_column for a reliable timestamp display
                                timestamp_to_display = row.get(sort_column, 0)
                                st.image(image, caption=f"Screenshot at {pd.to_datetime(timestamp_to_display, unit='s').strftime('%Y-%m-%d %H:%M:%S')}", use_column_width=True)
                            except Exception as e:
                                st.warning(f"Could not load image: {e}")
                        else:
                            st.caption("No screenshot available.")

        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
            st.exception(e) # Provides a more detailed traceback for debugging

    elif app_mode == "Search & Explore":
        # This view remains largely the same but with added safety checks
        st.header("Search Your Activity")
        query_text = st.text_input("Enter your search query:", placeholder="e.g., working on the logger script in vscode")
        
        if query_text:
            num_results = st.slider("Number of search results to return:", 1, 20, 5)
            try:
                results = collection.query(query_texts=[query_text], n_results=num_results, include=["metadatas", "distances"])
                if not results['ids'][0]:
                    st.warning("No results found for your query.")
                    return

                st.subheader("Search Results")
                for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                    with st.container(border=True):
                        st.markdown(f"**Result {i+1}** (Similarity Score: {1 - distance:.2f})")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Window:** `{metadata.get('window', 'N/A')}`")
                            st.markdown(f"**Text Summary:**")
                            st.info(metadata.get('text_summary', 'N/A'))
                            st.markdown(f"**Vision Summary:**")
                            st.success(metadata.get('vision_summary', 'N/A'))
                            with st.expander("Show Full Metadata"):
                                st.json(metadata)

                        with col2:
                            screenshot_path = metadata.get('screenshot_path')
                            if screenshot_path and screenshot_path != 'N/A' and os.path.exists(screenshot_path):
                                image = Image.open(screenshot_path)
                                timestamp = metadata.get('end_timestamp') or metadata.get('timestamp', 0)
                                st.image(image, caption=f"Screenshot at {pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')}", use_column_width=True)
                            else:
                                st.caption("No screenshot available.")
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")

if __name__ == "__main__":
    main()