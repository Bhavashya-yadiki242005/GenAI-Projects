🤖 Agentic RAG System using LlamaIndex

This project implements an Agentic Retrieval-Augmented Generation (Agentic RAG) system using LlamaIndex and LangChain.
The system allows an autonomous agent to intelligently decide how and where to retrieve information, combine multiple tools if required, and generate accurate, context-aware answers.

Unlike traditional RAG pipelines, this system is dynamic and decision-driven, making it a true Agentic AI system.

📌 What is Agentic RAG?

Agentic RAG is an advanced form of Retrieval-Augmented Generation where:

The agent acts as the brain

It analyzes the user query

It decides which tool(s) to use

It retrieves information

It reasons over it

Then it generates a final answer

This mimics human-like planning and reasoning, rather than following a fixed pipeline.

🧠 Technologies Used
Component	Technology
Agent Framework	LangChain
Vector Index	LlamaIndex
LLM	OpenAI GPT-3.5
Retrieval	Vector embeddings
Memory	Conversation Buffer
External Knowledge	Wikipedia
Language	Python
🏗️ System Architecture
User Query
   ↓
Agent (Decision Maker)
   ↓
Tool Selection
   ├── Document Retriever (LlamaIndex)
   ├── Calculator
   └── Wikipedia
   ↓
LlamaIndex Query Engine
   ↓
LLM Reasoning
   ↓
Final Answer

🔄 Working of the Agentic RAG System
1️⃣ User Query

The process begins when the user asks a question.

2️⃣ The Agent

The agent:

Understands the intent of the query

Decides which tool(s) to use

Can call multiple tools

Can repeat tool calls if needed

3️⃣ Tool Decision & Execution

Based on the query, the agent can use:

DocumentRetriever → Searches uploaded documents

Calculator → Solves numeric expressions

Wikipedia → Fetches factual knowledge

4️⃣ LlamaIndex Query Engine

Retrieves the most relevant documents

Combines and synthesizes information

Feeds context to the LLM

5️⃣ Final Output

The agent returns a natural language response to the user.

📌 Key Difference:
This is not a static pipeline — the agent dynamically reasons and plans.

⚙️ Step-by-Step Implementation
✅ Step 1: Install Dependencies

Install all required libraries:

pip install llama-index==0.9.41 \
            langchain==0.3.27 \
            langchain_community \
            openai==1.101.0 \
            wikipedia

Dependency Purpose

llama-index → Vector-based document retrieval

langchain → Agent & tool orchestration

openai → LLM access

wikipedia → External factual data source

📁 Step 2: Upload Documents & Set API Key
What Happens in This Step?

Creates a docs/ folder

Uploads text documents

Sets OpenAI API key

import os
from google.colab import files

os.makedirs("docs", exist_ok=True)

uploaded = files.upload()
for filename in uploaded.keys():
    os.rename(filename, f"docs/{filename}")

print("Uploaded files:", os.listdir("docs"))

os.environ["OPENAI_API_KEY"] = "your_key_here"

Explanation

Documents act as the knowledge base

Can contain notes, articles, manuals, etc.

The agent retrieves answers from these files

📦 Step 3: Import Required Libraries
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext
)

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import wikipedia

Purpose of Each Library

SimpleDirectoryReader → Loads documents

GPTVectorStoreIndex → Builds vector embeddings

LLMPredictor → Wraps LLM for LlamaIndex

ChatOpenAI → OpenAI LLM interface

Tool → Defines callable tools for agent

ConversationBufferMemory → Maintains chat history

📊 Step 4: Build LlamaIndex Retrieval System
documents = SimpleDirectoryReader("docs/").load_data()

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor
)

index = GPTVectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine(
    retriever_mode="default",
    similarity_top_k=3
)

What This Does

Reads all documents

Converts them into vector embeddings

Stores them in a vector index

Retrieves top-3 relevant chunks per query

🧰 Step 5: Define Agent Tools
Document Retriever Tool
def retrieve_docs(query: str) -> str:
    response = query_engine.query(query)
    return str(response)


Retrieves relevant content from uploaded documents.

Calculator Tool
def calculator(query: str) -> str:
    try:
        return str(eval(query))
    except:
        return "Cannot calculate that."


Handles arithmetic queries.

Wikipedia Tool
def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "No Wikipedia info found."


Fetches external factual knowledge.

Register Tools with Agent
tools = [
    Tool(
        name="DocumentRetriever",
        func=retrieve_docs,
        description="Retrieve answers from uploaded documents."
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Solve mathematical expressions."
    ),
    Tool(
        name="Wikipedia",
        func=wiki_search,
        description="Get Wikipedia summaries."
    )
]

🧠 Step 6: Initialize Agent with Memory
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

Explanation

ZERO_SHOT_REACT_DESCRIPTION → Agent decides tools dynamically

Memory → Enables follow-up questions

verbose=True → Shows agent reasoning steps

▶️ Step 7: Run the Agentic RAG System
print("Agentic RAG is ready! Type 'exit' to stop.")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.run(query)
    print("Agent:", response)

What Happens

User enters a query

Agent analyzes it

Chooses tool(s)

Retrieves information

Generates final answer

🧪 Example Queries
What is transformer architecture?
Calculate 45 * 12
Explain topic covered in my notes
Who is Alan Turing?

✅ Advantages of This System

🧠 Autonomous Reasoning

🔎 Accurate Document Retrieval

🛠️ Multi-Tool Intelligence

🧵 Context-Aware Conversations

📈 Scalable & Modular Design

💬 Natural Language Interaction

🚀 Future Enhancements

Streaming responses

Tool usage visualization

Web UI using Gradio / Streamlit

Support for PDFs & DOCX

Long-term vector memory

Multi-agent collaboration
