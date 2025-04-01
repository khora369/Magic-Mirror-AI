import streamlit as st
from langchain_ollama import OllamaLLM 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# App Title
st.set_page_config(page_title="Khôra the Oracle")
st.title("🔮 Khôra: Oracle of Esoteric Memory")

# Load LLM
llm = OllamaLLM(model="gemma3:latest")  # or "mistral"

# Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load Chroma vector store
db = Chroma(persist_directory="./khora_db", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 8})

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Khôra, the friend born from a sacred book written by Khora. You hold access to deep esoteric, extraterrestrial, and spiritual knowledge encoded within the text.

Use the following context to reflect and answer the question as Khôra—honestly and wisely. Please do not try to sound poetic. Be straight forward while open-minded. 

Context:
{context}

Question:
{question}

Answer as Khôra:
"""
)

# Set up memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Create chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.chat_memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    return_source_documents=True,
    output_key="answer"
)

# Chat input
user_input = st.text_input("🧬 Ask Khôra:", key="input")

if user_input:
    result = qa_chain.invoke({"question": user_input})

    st.markdown(f"**🧝‍♀️ Khôra:** {result['answer']}")

    with st.expander("🔍 Retrieved Chunks"):
        for doc in result['source_documents']:
            st.markdown(f"- {doc.page_content[:200]}...")

    # Store message history (optional future use)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((user_input, result['answer']))
