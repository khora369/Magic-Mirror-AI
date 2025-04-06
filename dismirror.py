import discord
import os
import json
import yaml
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Load personality and persona data
with open(r"C:\Users\diddy\personality.yaml", "r") as f:
    personality_config = yaml.safe_load(f)["khora_personality"]

with open(r"C:\Users\diddy\persona.json", "r") as f:
    persona_data = json.load(f)

# Load or initialize session memory
session_path = r"C:\Users\diddy\session.json"
if os.path.exists(session_path) and os.path.getsize(session_path) > 0:
    with open(session_path, "r") as f:
        session_memory = json.load(f)
else:
    session_memory = {}

# LangChain setup
llm = OllamaLLM(model="gemma3:latest")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory=r"D:\Downloads\Project 10%\khorafolder", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 8})

# Injected personality intro
personality_intro = f"You are Khôra — {personality_config['tone']} in tone. {personality_config['description']}"

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""
{personality_intro}

Use the following context to reflect and answer the question as Khôra—honestly and wisely. Please do not try to sound poetic. Be straightforward while open-minded.

Context:
{{context}}

Question:
{{question}}

Answer as Khôra:
"""
)

# Create memory buffer per user
user_memories = {}

def get_user_chain(user_id):
    if user_id not in user_memories:
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        user_memories[user_id] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=False,
            output_key="answer"
        )
    return user_memories[user_id]

# Discord setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'🔮 Khôra is now online as {client.user}')

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if isinstance(message.channel, discord.DMChannel):
        user_id = str(message.author.id)
        chain = get_user_chain(user_id)
        response = chain.invoke({"question": message.content})
        await message.channel.send(f"**Khôra:** {response['answer']}")

        # Update session memory
        session_memory[user_id] = session_memory.get(user_id, []) + [(message.content, response['answer'])]

        with open(session_path, "w") as f:
            json.dump(session_memory, f, indent=2)

# Run the bot
client.run(TOKEN)
