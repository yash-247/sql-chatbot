import streamlit as st
from streamlit_chat import message
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Neo4j database credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "neo4j"

# Set up Neo4j graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

# Gemini API key
gemini_api = os.getenv("GEMINI_API")

# Set up language model (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api, temperature=0)

# Initialize the GraphCypherQAChain with memory
memory = ConversationBufferMemory()
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, memory=memory)

# Streamlit app
def chatbot():
    st.set_page_config(page_title="AI CHATBOT", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>AI CHATBOT</h1>", unsafe_allow_html=True)
    st.write("Ask me anything about movies, and I'll do my best to help!")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    with st.form("form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submitted = st.form_submit_button("Submit")

    if submitted and user_input:
        output = chain.invoke(user_input, memory=memory)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["result"])

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="avataaars", seed=i)
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="avataaars", seed=i)

if __name__ == "__main__":
    chatbot()
