from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings  # Or use from langchain_community
from langchain_chroma import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

import streamlit as st
import os

# ✅ Add your Hugging Face token (keep this secret or use env vars later)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# ✅ Initialize the LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",  # or try "tiiuae/falcon-7b-instruct"
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# ✅ Embeddings & Vectorstore
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory="./khora_db", embedding_function=embedding)
retriever = db.as_retriever()

# ✅ Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ✅ Streamlit UI
st.title("🔮 Ask Khôra")
user_input = st.text_input("🧬 You:", "")

if user_input:
    result = qa_chain.invoke({"question": user_input})
    st.markdown(f"**Khôra:** {result['answer']}")
