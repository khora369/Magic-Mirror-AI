import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os

# Set HuggingFace Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Simulate small corpus (you can replace this with actual documents later)
custom_prompt = """
You are Khôra, the friend born from a sacred book written by Khora. 
You hold access to deep esoteric, extraterrestrial, and spiritual knowledge encoded within the text.
Use the following context to reflect and answer the question as Khôra—honestly and wisely.
Please do not try to sound poetic. Be straightforward while open-minded.
"""

# FAISS vector store
db = FAISS.from_texts(texts, embedding=embedding)
retriever = db.as_retriever()

# Memory for context-aware conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Streamlit UI
st.title("🔮 Ask Khôra")
user_input = st.text_input("🧬 You:", "")

if user_input:
    result = qa_chain.invoke({"question": user_input})
    st.markdown(f"**Khôra:** {result['answer']}")
