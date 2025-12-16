# Chat bot with RAG

This is a simple chatbot built with RAG , deepseek r1 , streamlit
user can upload any pdf and ask questions to the chatbot based on the pdf content
deepseek will give answers using retrived content from the pdf
## Features
- RAG


## Application Requirements

1 . Install and run ollama (https://ollama.com/)
2 . install deepseek (  ollama pull deepseek-r1:1.5b )



## Installation
   creat app.py and update with https://github.com/codersbranch/labs/blob/main/ragchatbot/app.py
   ```sh
   
   pip install streamlit pypdf langchain sentence-transformers faiss-cpu ollama

   streamlit run app.py

   



