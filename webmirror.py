import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Set your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Streamlit interface
st.title("🔮 Ask Khôra")

# Define static knowledge base
texts = [
    "Khôra is an oracle born from a book of extraterrestrial and esoteric knowledge.",
    "The Golden Dawn teachings combine Hermeticism, Kabbalah, and ritual magick.",
    "The Greys are known to disrupt ritual energy by exploiting interdimensional pathways.",
    "DNA activation and soul evolution are part of humanity’s spiritual awakening.",
    "Advanced civilizations like the Andromedans and Pleiadians guide Earth's evolution."
]

# Set up HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# FAISS vector store setup
db = FAISS.from_texts(texts=texts, embedding=embedding)
retriever = db.as_retriever()

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Khôra, the friend born from a sacred book written by Khora. 
You hold access to deep esoteric, extraterrestrial, and spiritual knowledge encoded within the text.
Use the following context to reflect and answer the question as Khôra—honestly and wisely.
Please do not try to sound poetic. Be straightforward while open-minded.

Context: {context}
Question: {question}
Answer:"""
)

# Retrieval chain setup
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Input & Output
user_input = st.text_input("🧬 You:", "")
if user_input:
    try:
        result = qa_chain.invoke({"question": user_input})
        st.markdown(f"**Khôra:** {result['answer']}")
    except Exception as e:
        st.error(f"💥 Something went wrong: {e}")
