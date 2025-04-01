import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

# Set your Hugging Face token here or use st.secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load model from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",  # or try "tiiuae/falcon-7b-instruct"
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load your database (in-memory example)
from langchain_community.vectorstores import FAISS

db = FAISS.from_texts(["This is Khôra, your oracle AI."], embedding=embedding)
retriever = db.as_retriever()

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Q&A Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Streamlit app UI
st.title("🔮 Ask Khôra")
user_input = st.text_input("🧬 You:", "")

if user_input:
    result = qa_chain.invoke({"question": user_input})
    st.markdown(f"**Khôra:** {result['answer']}")
