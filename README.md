
# LifeLog-AI

**A local, multimodal AI that chronicles your digital activity on Windows, enabling you to query your own history with a personal, privacy-first RAG pipeline.**

---

## Core Concept

LifeLog-AI is a personal research project designed to create a private, local "digital memory" or "exocortex." By passively logging user interactions—from keystrokes and mouse clicks to contextual screenshots—it builds a rich, searchable database of your own digital life.

The entire ecosystem is built on a philosophy of **local-first, privacy-centric AI**. All data is stored and processed on your machine, and it leverages small, efficient open-source language models via Ollama. This ensures that your personal activity data remains completely private and under your control.

## Features

-   **Passive & Efficient Logging:** A lightweight, high-performance logger (`logger.py`) runs in the background, capturing keystrokes, clicks, and the active window.
-   **Intelligent Multimodal Capture:** Screenshots are not taken randomly. They are triggered by significant context shifts, such as changing applications, ensuring relevance while respecting system resources.
-   **Smart Local Processing:** A resilient batch processor (`batch_processor.py`) converts raw, noisy logs into "chunks" of meaningful activity.
-   **Visual Deduplication:** Saves time and compute by using perceptual hashing to avoid re-analyzing screenshots that are visually identical.
-   **Personal RAG Pipeline:** Use a command-line interface (`main.py`) to ask natural language questions about your past activity and receive AI-generated answers with source context.
-   **Live Streaming Answers:** The query interface streams answers token-by-token for a responsive, modern user experience.
-   **Visual Dashboard:** An interactive web dashboard (`dashboard.py`) built with Streamlit allows for visual exploration and semantic search of your activity history.
-   **100% Local & Free:** The entire stack, from the AI models (Ollama) to the vector database (ChromaDB), runs locally and is built with free, open-source software.

## Technology Stack

-   **Core AI & LangChain:**
    -   `LangChain`: The primary framework for building the RAG pipeline.
    -   `Ollama`: Serves local LLMs.
    -   `phi` (Text Model): A small, fast LLM for text generation and summarization.
    -   `llava` / `moondream` (Vision Models): Multimodal LLMs for analyzing screenshots.
-   **Data & Storage:**
    -   `SQLite`: Stores the raw, chronological interaction logs.
    -   `ChromaDB`: A local vector database for storing and searching activity embeddings.
    -   `Sentence-Transformers`: For creating the vector embeddings.
-   **User Interaction & System:**
    -   `pynput`: Captures keyboard and mouse events.
    -   `pywin32`: Retrieves active window information on Windows.
    -   `mss` & `Pillow`: For fast, efficient screenshot capture and optimization.
    -   `ImageHash`: For perceptual hashing and image deduplication.
-   **Frontend & Visualization:**
    -   `Streamlit`: Powers the interactive web dashboard for visual exploration.

## System Architecture

The ecosystem follows a simple, robust data flow:

```
[Windows Environment]
       |
       | (Keystrokes, Clicks, Screen)
       v
+--------------+      +--------------------------+
|  logger.py   |----->|  user_interactions.db    | (Raw SQLite Logs)
+--------------+      |  /images/                | (Optimized Screenshots)
                      +--------------------------+
                                 |
                                 | (Raw data is read by...)
                                 v
                      +--------------------------+
                      |   batch_processor.py     |
                      | (Chunks, Summarizes,     |
                      |  Analyzes, Embeds)       |
                      +--------------------------+
                                 |
                                 | (Embeddings & Metadata are stored in...)
                                 v
                      +--------------------------+
                      |       chroma_db          | (Vector Database)
                      +--------------------------+
                         ^                ^
                         | (Searched by)  | (Searched by)
                         |                |
                +--------------+    +---------------+
                |   main.py    |    | dashboard.py  |
                | (RAG Query)  |    | (Visual UI)   |
                +--------------+    +---------------+
```

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/LifeLog-AI.git
    cd LifeLog-AI
    ```

2.  **Create Directories:**
    Create the necessary folders in the project root:
    ```bash
    mkdir data
    mkdir images
    ```

3.  **Install Ollama:**
    Download and install Ollama for Windows from [ollama.com](https://ollama.com/download).

4.  **Pull the AI Models:**
    Open a terminal and pull the required models:
    ```bash
    ollama pull phi
    ollama pull llava
    # ollama pull moondream (Optional, faster alternative)
    ```

5.  **Set Up Python Environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

6.  **Install Dependencies:**
    Install all required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You would need to create a `requirements.txt` file by running `pip freeze > requirements.txt`)*

## Usage Workflow

The system is designed to be used in a simple, cyclical manner.

1.  **Run the Logger:** Start the logger to capture your activity. It will run silently in the background.
    ```bash
    python logger.py
    ```
    Press `Ctrl+C` in its terminal to stop it gracefully.

2.  **Process Your Data:** When you're ready to update your knowledge base (e.g., at the end of the day), run the batch processor.
    ```bash
    python batch_processor.py
    ```

3.  **Ask Questions:** Use the main query interface to talk to your digital memory.
    ```bash
    python main.py
    ```

4.  **Explore Visually:** Use the Streamlit dashboard for a web-based view of your data.
    ```bash
    streamlit run dashboard.py
    ```

## Ethical Considerations

**This is a powerful tool designed for personal self-analysis ONLY.**
-   **Privacy:** The data collected is extremely sensitive. The local-first design is intentional to ensure you are the sole controller of your data. Never run this tool on a computer that is not your own.
-   **Transparency:** You are both the researcher and the subject. Be mindful of what you are collecting and for what purpose.
-   **Security:** Ensure your local data is secure. The database contains a detailed log of your digital life.

## Future Work

-   Integrate `LangSmith` for more detailed tracing and evaluation of the RAG pipeline.
-   Implement `LangGraph` to model more complex, stateful analysis agents.
-   Develop more sophisticated text parsing to better handle code, commands, and prose.
-   Create a long-term memory summarization agent that periodically condenses old chunks.
