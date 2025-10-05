# LifeLog-AI

**A local, multimodal AI that chronicles your digital activity on Windows, enabling you to query your own history with a personal, privacy-first RAG pipeline and predictive ML.**

---

## Project Status

**Active Development & Research:** This project is a fully functional, end-to-end ecosystem for personal data analysis. All core features are implemented. Future work is focused on refining the predictive models and exploring more advanced agentic behavior.

## Core Concept

LifeLog-AI is a personal research project designed to create a private, local "digital memory" or "exocortex." By passively logging user interactions—from keystrokes and mouse clicks to contextual screenshots—it builds a rich, searchable database of your own digital life.

The entire ecosystem is built on a philosophy of **local-first, privacy-centric AI**. All data is stored and processed on your machine, and it leverages small, efficient open-source language models via Ollama. This ensures that your personal activity data remains completely private and under your control.

## Key Features

-   **Passive & Efficient Logging:** A lightweight, high-performance logger (`logger.py`) runs in the background, capturing user interactions with minimal system impact.
-   **Intelligent Multimodal Capture:** Screenshots are triggered by significant context shifts (like changing applications) to provide relevant visual context.
-   **Automated Data Pipeline:** A unified processor (`processor.py`) handles the entire data lifecycle:
    -   **Smart Chunking:** Converts raw, noisy logs into meaningful chunks of activity.
    -   **Multimodal Analysis:** Uses a VLM (`llava`) for visual descriptions and an OCR engine (`Tesseract`) to read on-screen text.
    -   **Visual Deduplication:** Saves time and compute by using perceptual hashing to avoid re-analyzing identical screenshots.
-   **Deep Learning & NLP Analysis:** The pipeline automatically runs advanced analysis on your entire history:
    -   **Topic Modeling (`BERTopic`):** Discovers and labels the underlying topics of your work.
    -   **Task Clustering (`HDBSCAN`):** Identifies your recurring patterns of behavior.
    -   **Anomaly Detection:** Finds your most unusual or outlier activities.
    -   **Predictive Model Training:** Automatically trains a Deep Learning model (`MLPClassifier`) to recognize your tasks.
-   **Multiple Interfaces:**
    -   **RAG Query Interface (`main.py`):** Ask natural language questions and receive live, streaming, color-coded answers from an AI that has studied your history.
    -   **All-in-One Utility Dashboard (`utils_dashboard.py`):** A fast, lightweight Tkinter GUI to visually explore data, manage databases, and run the **Live Task Monitor**.

## Technology Stack

-   **Core AI & Orchestration:**
    -   `LangChain`: Framework for the RAG pipeline.
    -   `Ollama`: Serves local LLMs.
    -   `phi` (Text LLM): For generating answers in the RAG pipeline.
    -   `llava` (Vision LLM): For screenshot summarization.
-   **Data Processing & Storage:**
    -   `SQLite`: Stores raw, chronological interaction logs.
    -   `ChromaDB`: Local vector database for activity embeddings.
    -   `Sentence-Transformers`: Creates vector embeddings from text.
    -   `Tesseract` & `pytesseract`: For Optical Character Recognition (OCR).
-   **Machine Learning & NLP:**
    -   `scikit-learn`: For clustering, anomaly detection, and the predictive MLP model.
    -   `BERTopic`: For state-of-the-art topic modeling.
    -   `spaCy`: For Named Entity Recognition (NER).
    -   `UMAP` & `HDBSCAN`: For advanced clustering and dimensionality reduction.
-   **System & UI:**
    -   `pynput` & `pywin32`: For capturing user interactions on Windows.
    -   `mss`, `Pillow`, `ImageHash`: For efficient screenshot handling.
    -   `Tkinter`: Powers the fast, native all-in-one utility dashboard.
    -   `FPDF2`: For generating structured PDF analysis reports.
    -   `colorama`: For a color-coded terminal experience.

## The LifeLog-AI Workflow

The system is designed to be used in a simple, structured cycle. **You must run the steps in this order.**

### **Step 1: Run the Logger**

Start the logger to capture your activity. It will run silently in the background. It is the only script you need running while you work.

```bash
python logger.py
```
*(Press `Ctrl+C` in its terminal to stop it gracefully.)*

### **Step 2: Run the Processor**

This is the main "update" command. When you're ready to analyze your newly logged data, run this single script. It will automatically process new data, then re-analyze your entire history, generate reports, and train a new predictive model.

```bash
python processor.py
```
*(This may take a significant amount of time, as it is performing all the AI/ML analysis.)*

### **Step 3: Interact with Your Data**

Once the processor has finished, your knowledge base and predictive model are up to date. You can now use the interface tools.

-   **To Ask Questions:**
    ```bash
    python main.py
    ```

-   **To Use the Full GUI Dashboard:**
    This is the recommended tool for live monitoring and database management.
    ```bash
    python utils_dashboard.py

    ```
## Testing the video
[Watch the full demo video](demo/lifelogai.mp4)

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/LifeLog-AI.git
    cd LifeLog-AI
    ```

2.  **Install System Dependencies:**
    -   **Ollama:** Download and install from [ollama.com](https://ollama.com/download).
    -   **Tesseract OCR:** Install from the [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki). **Remember to add it to your system PATH.**

3.  **Create Directories:**
    Create the necessary folders in the project root:
    ```bash
    mkdir data images outputs fonts
    ```

4.  **Download Fonts:**
    Download the DejaVu fonts from [dejavu-fonts.github.io](https://dejavu-fonts.github.io/), and place `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` into the `fonts` folder.

5.  **Pull AI Models:**
    ```bash
    ollama pull phi
    ollama pull llava
    ollama pull moondream ( use this instead of llava in case of a weaker system)
    ```

6.  **Set Up Python Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

7.  **Install Python Dependencies:**
    Create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment after installing all packages mentioned in our steps. Then, you or others can install from it:
    ```bash
    pip install -r requirements.txt
    ```

## Ethical Considerations

**This is a powerful tool designed for personal self-analysis ONLY.**
-   **Privacy:** The data collected is extremely sensitive. The local-first design is intentional to ensure you are the sole controller of your data.
-   **Transparency:** You are both the researcher and the subject. Be mindful of what you are collecting.
-   **Security:** Ensure your local data is secure. The database contains a detailed log of your digital life.
