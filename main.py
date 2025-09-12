import os
import webbrowser
import chromadb # Import the native ChromaDB client
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# --- Configuration ---
CHROMA_PATH = os.path.join("data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_MODEL = "phi"
COLLECTION_NAME = "user_activity_collection"

# --- Initialize Components ---
print("Loading the knowledge base...")
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
llm = Ollama(model=TEXT_MODEL, callbacks=[StreamingStdOutCallbackHandler()])

# --- NEW: Initialize the NATIVE ChromaDB Client (The proven method from the dashboard) ---
try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print("Native ChromaDB client connected successfully.")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to connect native ChromaDB client: {e}")
    exit()

# --- NEW, 100% RELIABLE RETRIEVER LOGIC ---
def retrieve_documents(query: str) -> list[Document]:
    """
    Uses the native chromadb client to perform the search, guaranteeing it works.
    """
    # The native query returns a dictionary of lists.
    results = collection.query(
        query_texts=[query],
        n_results=2,
        include=["metadatas", "documents"]
    )
    
    # Manually reconstruct the LangChain Document objects from the raw results.
    final_docs = []
    # The results are nested in a list, so we access the first element [0]
    for doc_content, metadata in zip(results['documents'][0], results['metadatas'][0]):
        reconstructed_doc = Document(
            page_content=doc_content,
            metadata=metadata
        )
        final_docs.append(reconstructed_doc)
    
    return final_docs

# --- Prompt and Chain (No changes needed here) ---
template = """
You are an AI assistant analyzing a user's computer activity logs.
Answer the question based ONLY on the context provided below.
If the context doesn't contain the answer, just say that you don't have enough information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"Context Chunk:\n{doc.page_content}\n(Source Screenshot: {doc.metadata.get('screenshot_path', 'N/A')})"
        for doc in docs
    )

rag_chain = (
    {"context": RunnableLambda(lambda x: retrieve_documents(x["question"])) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Main Application Loop ---
if __name__ == "__main__":
    print("\nKnowledge base loaded. You can now ask questions about your activity.")
    print("   Type 'exit' to quit.")
    print("-" * 50)

    while True:
        try:
            query = input("\nAsk your question: ")
            if query.lower() == 'exit':
                break
            if not query.strip():
                continue

            print("\nFinding relevant moments in your history...")
            relevant_docs = retrieve_documents(query)

            if not relevant_docs:
                print("No relevant information found in your history.")
                continue

            print("\n--- Top Context Found ---")
            for i, doc in enumerate(relevant_docs):
                print(f"[{i+1}] Summary: {doc.page_content[:200]}...")
                print(f"    Screenshot: {doc.metadata.get('screenshot_path', 'N/A')}\n")

            print("--- AI Answer (Streaming) ---")
            _ = rag_chain.invoke({"question": query})
            
            print("\n" + "-" * 20)

            while True:
                choice = input("Enter a number to open its screenshot, or press Enter to continue: ")
                if choice.isdigit() and 1 <= int(choice) <= len(relevant_docs):
                    path = relevant_docs[int(choice) - 1].metadata.get('screenshot_path')
                    if path and path != 'N/A' and os.path.exists(path):
                        webbrowser.open(f'file://{os.path.realpath(path)}')
                    else:
                        print("Screenshot not available or path is invalid.")
                elif choice == "":
                    break
                else:
                    print("Invalid input.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")