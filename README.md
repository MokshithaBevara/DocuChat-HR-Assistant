# 📄 DocuChat HR Assistant

An AI-powered HR assistant that helps employees quickly find information from company HR policies and handbooks.

## Features

* HR policy question answering
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database
* Semantic search using embeddings
* Groq LLM integration
* Streamlit-based chat interface

## Tech Stack

* Python
* Streamlit
* ChromaDB
* Sentence Transformers
* Groq API

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Overview

This project converts HR handbook content into vector embeddings and stores them in ChromaDB. When a user asks a question, the most relevant policy sections are retrieved and provided to the language model, enabling accurate and context-aware responses.

## Author

Mokshitha Bevara
