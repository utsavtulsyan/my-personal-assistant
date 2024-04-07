# My personal assistant
This is a simple implementation of a personal assistant that can read all the documents and answer queries on it.

It is safe to use with confidential data, as this does not send your data to any thir party server, all your data is processed locally.

> DISCLAIMER: It is very prone to hallucination, and context muddling. Only for demonstration purpose. No warranty or guarantee.

# Tech stack

1. Ollama - llama2 for local LLM inference and embedding
2. Python
3. FAISS for storing and searching indexes

# Setup
make sure llama2 is installed in the ollama environment and running on default port
```sh
pip install pipenv streamlit

pipenv shell

pipenv install
```

# Usage
1. To load files: create a folder named `mydocs` place all files inside it
2. Run command to index all files
```sh
python file_loader.py
```
3. Start chat interface

```sh
streamlit run chat.py
```