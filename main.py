import os
import webbrowser
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import colorama

# --- Colorama Setup ---
colorama.init(autoreset=True)
C = {
    "query": colorama.Fore.CYAN,
    "info": colorama.Fore.YELLOW,
    "header": colorama.Fore.GREEN + colorama.Style.BRIGHT,
    "context": colorama.Fore.WHITE,
    "error": colorama.Fore.RED,
    "reset": colorama.Style.RESET_ALL
}

# --- Configuration ---
CHROMA_PATH = os.path.join("data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_MODEL = "phi"
COLLECTION_NAME = "user_activity_collection"

# --- Initialize Components ---
print(f"{C['info']}Loading the knowledge base...")
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
llm = Ollama(model=TEXT_MODEL, callbacks=[StreamingStdOutCallbackHandler()])

try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"{C['info']}Native ChromaDB client connected successfully.")
except Exception as e:
    print(f"{C['error']}[CRITICAL ERROR] Failed to connect native ChromaDB client: {e}")
    exit()

# --- Reliable Retriever Logic ---
def retrieve_documents(query: str) -> list[Document]:
    """Uses the native chromadb client to perform the search."""
    # Using the query transformation we developed for better precision
    transformed_query = f"A user activity log chunk about: {query}"
    print(f"{C['info']}INFO: Transformed query to: '{transformed_query}'")
    
    results = collection.query(
        query_texts=[transformed_query],
        n_results=3, # Retrieve top 3 results
        include=["metadatas", "documents"]
    )
    
    final_docs = []
    if results and results['ids'][0]:
        for doc_content, metadata in zip(results['documents'][0], results['metadatas'][0]):
            reconstructed_doc = Document(page_content=doc_content, metadata=metadata)
            final_docs.append(reconstructed_doc)
    
    return final_docs

# --- Prompt and Chain ---
template = """
You are a helpful AI assistant analyzing a user's computer activity logs.
Answer the user's question based ONLY on the context provided below.
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
    print(f"\n{C['header']}Knowledge base loaded. You can now ask questions about your activity.")
    print(f"{C['header']}   Type 'exit' to quit.")
    print(f"{C['header']}" + "-" * 50)

    while True:
        try:
            query = input(f"\n{C['query']}Ask your question: {C['reset']}")
            if query.lower() == 'exit':
                break
            if not query.strip():
                continue

            print(f"\n{C['info']}Finding relevant moments in your history...")
            relevant_docs = retrieve_documents(query)

            if not relevant_docs:
                print(f"{C['error']}No relevant information found in your history.")
                continue

            print(f"\n{C['header']}--- Top Context Found ---")
            for i, doc in enumerate(relevant_docs):
                print(f"{C['context']}[{i+1}] Summary: {doc.page_content[:250]}...")
                print(f"{C['context']}    Screenshot: {doc.metadata.get('screenshot_path', 'N/A')}\n")

            print(f"{C['header']}--- AI Answer (Streaming) ---")
            
            # --- NEW: Change terminal color for the LLM response ---
            if os.name == 'nt': # This command only works on Windows
                os.system('color c') 
            
            _ = rag_chain.invoke({"question": query})
            
            # --- NEW: Revert terminal color back to default ---
            if os.name == 'nt':
                os.system('color 07')

            print("\n" + C['header'] + "-" * 20)

            while True:
                choice_prompt = f"{C['query']}Enter a number to open its screenshot, or press Enter to continue: {C['reset']}"
                choice = input(choice_prompt)
                if choice.isdigit() and 1 <= int(choice) <= len(relevant_docs):
                    path = relevant_docs[int(choice) - 1].metadata.get('screenshot_path')
                    if path and path != 'N/A' and os.path.exists(path):
                        print(f"{C['info']}Opening {path}...")
                        webbrowser.open(f'file://{os.path.realpath(path)}')
                    else:
                        print(f"{C['error']}Screenshot not available or path is invalid.")
                elif choice == "":
                    break
                else:
                    print(f"{C['error']}Invalid input.")

        except Exception as e:
            print(f"\n{C['error']}An error occurred: {e}")
            if os.name == 'nt':
                os.system('color 07') # Ensure color is reset on error