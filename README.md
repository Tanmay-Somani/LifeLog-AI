# Cognitive Chronicle

**A local, multimodal AI that chronicles your digital activity on Windows, enabling you to query your own history with a personal, privacy-first RAG pipeline.**

---

## Core Concept

Cognitive Chronicle is a personal research project designed to create a private, local "digital memory" or "exocortex." By passively logging user interactions—from keystrokes and mouse clicks to contextual screenshots—it builds a rich, searchable database of your own digital life.

The entire ecosystem is built on a philosophy of **local-first, privacy-centric AI**. All data is stored and processed on your machine, and it leverages small, efficient open-source language models via Ollama. This ensures that your personal activity data remains completely private and under your control.

## Features

-   **Passive & Efficient Logging:** A lightweight, high-performance logger (`logger.py`) runs in the background, capturing keystrokes, clicks, and the active window.
-   **Intelligent Multimodal Capture:** Screenshots are not taken randomly. They are triggered by significant context shifts, such as changing applications or after a period of inactivity followed by a specific action, ensuring relevance while respecting system resources.
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
