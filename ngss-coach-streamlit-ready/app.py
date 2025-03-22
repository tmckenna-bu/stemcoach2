import streamlit as st
import openai
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pathlib import Path
import re

load_dotenv()

st.set_page_config(page_title="STEM EquityCoach", layout="wide")
st.title("👩‍🏫 STEM EquityCoach")
st.markdown("Your NGSS-aligned, curriculum-smart, teacher-supportive chatbot coach.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

INDEX_PATH = "faiss_index"
DATA_FILE = "cleaned_curriculum_chunks.jsonl"

@st.cache_resource
def build_vector_index():
    if not Path(DATA_FILE).exists():
        st.error("Curriculum data not found.")
        return None
    docs = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docs.append(Document(
                page_content=item['text'],
                metadata=item['metadata']
            ))
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def classify_intent(query):
    query = query.lower()
    if re.search(r'\b(ls[1-4]\.[a-d])\b', query):
        return "find_lessons_by_dci"
    if "phenomenon" in query:
        return "find_lessons_by_phenomenon"
    if "sep" in query or "science practice" in query:
        return "identify_seps"
    if "model" in query and "students" in query:
        return "modeling_support"
    if "assessment" in query:
        return "assessment_identification"
    return "general_support"

def generate_response(user_query, intent, docs):
    context_text = "\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    system_prompt = """You are a warm, knowledgeable curriculum coach helping science teachers implement the NGSS.
Your tone is supportive, collaborative, and encouraging. Respond with practical suggestions and
classroom wisdom based on curriculum materials provided."""

    if intent == "identify_seps":
        system_prompt += "\nFocus on identifying science practices and how students engage in them."
    elif intent == "modeling_support":
        system_prompt += "\nFocus on where students build, revise, or use models."
    elif intent == "assessment_identification":
        system_prompt += "\nFocus on identifying embedded assessments and what standards they target."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Teacher question: {user_query}\n\nCurriculum context:\n{context_text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.6
    )
    return response['choices'][0]['message']['content']

db = build_vector_index()
if db is None:
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your coach a question:", "")

if query:
    st.session_state.chat_history.append(("user", query))
    intent = classify_intent(query)
    results = db.similarity_search(query, k=3)
    reply = generate_response(query, intent, results)
    st.session_state.chat_history.append(("coach", reply))

for speaker, text in st.session_state.chat_history:
    if speaker == "user":
        st.markdown(f"**🧑‍🏫 You:** {text}")
    else:
        st.markdown(f"**🤖 Coach:** {text}")
