# E-Commerce FAQ Bot using LangGraph and ChromaDB

This project is an **E-Commerce FAQ Bot** built using LangGraph for agentic workflow orchestration and ChromaDB for vector-based retrieval.  
The bot can answer common customer queries about products, orders, shipping, returns, and other FAQs in an interactive way.

---

## Project Overview

- Uses LangGraph to define and manage an agentic graph of tools and LLM calls for FAQ resolution.
- Uses ChromaDB as a vector store to index and retrieve FAQ documents.
- Provides a Streamlit-based UI for users to chat with the bot.
- Designed as a capstone project for the Agentic AI Program.

---

## Live Demo

You can access the deployed app here:

🔗 **Deployment link:** https://agentic-ai-capstone-project.streamlit.app/

---

## Features

- Natural language Q&A over e-commerce FAQs.
- Retrieval-augmented responses using ChromaDB.
- Clean and simple Streamlit chat interface.
- Configurable to new FAQ datasets with minimal changes.
- Modular code structure (notebook + Streamlit app) for experimentation and deployment.

---

## Tech Stack

- **Language:** Python  
- **Frameworks/Libraries:**  
  - LangGraph (agentic workflows)  
  - ChromaDB (vector database)  
  - Streamlit (web UI)  
- **Environment:** Jupyter Notebook for development and experimentation

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bhanukumardev/Agentic-AI-Capstone-Project.git
cd Agentic-AI-Capstone-Project
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

If you have a `requirements.txt` (recommended), run:

```bash
pip install -r requirements.txt
```

Otherwise, install the key libraries manually:

```bash
pip install langgraph chromadb streamlit
```

(and any additional libraries used in `capstone_streamlit.py` or `day13_capstone.ipynb`).

### 4. Run the Streamlit app locally

```bash
streamlit run capstone_streamlit.py
```

Open the URL shown in the terminal (usually http://localhost:8501) to interact with the FAQ bot locally.

---

## Project Files

- `capstone_streamlit.py` – Streamlit app entry point for the E-Commerce FAQ Bot.
- `day13_capstone.ipynb` – Notebook with experimentation, pipeline development, and exploration.
- `.vscode/` – Editor configuration for development.

---

## How It Works (High Level)

1. **Data Ingestion:**  
   FAQ-style text data is embedded and stored in ChromaDB as vector representations.

2. **User Query:**  
   A user sends a question through the Streamlit interface.

3. **Retrieval:**  
   The system queries ChromaDB for the most relevant FAQ entries matching the user question.

4. **Agentic Orchestration (LangGraph):**  
   LangGraph coordinates retrieval and LLM reasoning steps to generate a helpful, context-aware response.

5. **Response:**  
   The answer is shown in the chat UI, optionally including retrieved FAQ snippets as context.

---

## Future Improvements

- Add support for multiple e-commerce domains (electronics, fashion, etc.).
- Integrate authentication for admin dashboards to upload/update FAQs.
- Add conversation history awareness for multi-turn customer support.
- Improve UI design and add analytics for FAQ coverage/usage.

---

## Author & Program Details

**Built by:** Bhanu Kumar Dev  
**Roll:** 2328162  
**Batch:** ExcelR & KIIT_Feb26_ Agentic AI Program _B7  

If you find this project useful or have suggestions, feel free to open an issue or create a pull request.

---

## License

This project is currently shared for educational and demonstration purposes.  
You may fork and experiment with it; please credit the original author when reusing substantial parts of the code.
