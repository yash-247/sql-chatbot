import streamlit as st
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'KEY HERE ' # API KEY HERE
# Initialize the Google Gemini LLM with the API key
gemini_api_key = "KEY HERE" # API KEY HERE

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embeddings(text):
    embeddings = model.encode([text])
    return embeddings[0]

# Example schema information
schema_info = [
    {"table": "order_item_refunds", "columns": ["order_item_refund_id", "order_item_id", "created_at", "order_id", "refund_amount_usd"], "description": "table name is 'order_item_refunds'Refunded order items details"},
    {"table": "order_items", "columns": ["order_item_id", "created_at", "order_id", "product_id", "is_primary_item", "price_usd", "cogs_usd"], "description": "table name is 'order_items'Order items details where order_item_id is auto incremented primary key, price_usd is price of item and cogs_usd is cost of goods sold, created_at is the date when order item is created"},
    {"table": "orders", "columns": ["order_id", "created_at", "website_session_id", "user_id", "primary_product_id", "items_purchased", "price_usd", "cogs_usd"], "description": "table name is 'orders'Order details where website_session_id is the session id of the user, items_purchased is the number of items purchased from the user_id"},
    {"table": "products", "columns": ["product_id", "created_at", "product_name"], "description": "table name is 'products'Product information where product_id 1 belongs to 'The Original Mr.Fuzzy', product_id 2 belongs to 'The Forever Love Bear', product_id 3 belongs to 'The Birthday Sugar Panda', product_id 4 belongs to 'The Hudson River Mini bear'"},
    {"table": "website_pageviews", "columns": ["website_pageview_id", "created_at", "website_session_id", "pageview_url"], "description": "table name is 'website_pageviews'Website Pageviews details with pageview_url as the url of the pageviewed, website_session_id is the session id of the user"},
    {"table": "website_sessions", "columns": ["website_session_id", "created_at", "user_id", "is_repeat_session", "utm_source", "utm_campaign", "utm_content", "device_type", "http_referer"], "description": "table name is 'website_sessions'Website session details containing utm source as 'gsearch', 'null', 'bsearch', 'socialbook', is_repeat_session is the repeat session of the user where 0 is not repeat session and 1 is a repeat session, utm_campaign is the campaign with values 'nonbrand', 'brand', 'null', 'pilot', 'desktop_targeted', device_type are of two types mobile or desktop"}
]

# Create FAISS index
dim = 384  # Dimensionality of the embeddings
index = faiss.IndexFlatIP(dim)
index_id_map = {}

# Convert schema information to embeddings and add to FAISS index
schema_embeddings = []
for idx, schema in enumerate(schema_info):
    embedding = generate_embeddings(schema['description'])
    schema_embeddings.append(embedding)
    index_id_map[idx] = schema['table']
    index.add(np.array([embedding]))

# Function to find relevant schemas dynamically
def find_relevant_schemas(user_query, relevance_threshold=0.3):
    query_embedding = generate_embeddings(user_query)
    distances, indices = index.search(np.array([query_embedding]), len(schema_info))
    
    relevant_schemas = []
    for i, distance in enumerate(distances[0]):
        if distance >= relevance_threshold:
            relevant_schemas.append({
                "table": index_id_map[indices[0][i]],
                "description": schema_info[indices[0][i]]["description"],
                "columns": schema_info[indices[0][i]]["columns"]
            })
        else:
            break
    return relevant_schemas

# Update the prompt template
template = """
Based on the table schemas below, write MySQL query that would answer the user's question. You will not generate new columns and tables other than the ones provided in the schema.when asked question on calculation based on last week use this created_at >= NOW() - INTERVAL 7 DAY and optimize accordingly.

Schemas:
{description}

Columns:
{columns}

Question: {question}

SQL Query:
"""

# Function to generate the prompt
def generate_prompt(question, relevant_schemas):
    schema_descriptions = []
    schema_columns = []
    
    for schema in relevant_schemas:
        schema_descriptions.append(schema["description"])
        schema_columns.append(", ".join(schema["columns"]))
    
    schema_descriptions_str = "\n".join(schema_descriptions)
    schema_columns_str = "\n".join(schema_columns)
    
    return template.format(description=schema_descriptions_str, columns=schema_columns_str, question=question)

prompt = ChatPromptTemplate.from_template(template)

db_uri = "mysql+mysqlconnector://root:12345678@localhost:3306/mavenfuzzyfactory"
db = SQLDatabase.from_uri(db_uri)

def get_schema(_):
    return db.get_table_info()


llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key, temperature=0, convert_system_message_to_human=True)

# Define a custom prompt function
def custom_prompt(prompt_string):
    messages = [HumanMessage(content=prompt_string)]
    result = llm(messages)
    clean_sql = result.content.strip().strip('```sql').strip('```')
    return clean_sql

# Execute the SQL query and return the results
def execute_sql_query(sql_query):
    result = db.run(sql_query)
    return result

# Define a function to create a human-readable message from the SQL result
def create_human_friendly_message(sql_result, question):
    prompt_string = f"Based on the SQL query result below, provide a human-friendly explanation for the following question:\n\nQuestion: {question}\n\nSQL Query Result: {sql_result},If the SQL Query doesn't return anything Display 'No results found' You cannot generate your own answers, you must use the SQL Query result to create the human-friendly message only."
    messages = [HumanMessage(content=prompt_string)]
    result = llm(messages)
    return result.content

# Memory for conversations
conversation_memory = []

def add_to_memory(user_query, generated_sql, sql_result, human_friendly_message):
    conversation_memory.append({
        "user_query": user_query,
        "generated_sql": generated_sql,
        "sql_result": sql_result,
        "human_friendly_message": human_friendly_message
    })

def display_conversation_memory():
    if conversation_memory:
        st.subheader("Conversation History")
        for idx, convo in enumerate(conversation_memory):
            st.write(f"**Query {idx + 1}:**")
            st.write("- User Query:", convo["user_query"])
            st.write("- Generated SQL:", convo["generated_sql"])
            st.write("- SQL Result:", convo["sql_result"])
            st.write("- Human-Friendly Message:", convo["human_friendly_message"])
            st.markdown("---")
    else:
        st.info("No conversations yet.")

def clear_memory():
    global conversation_memory
    conversation_memory = []
    st.info("Memory cleared.")

st.title("Conversational SQL Assistant")

user_query = st.text_input("Enter your query:")

if st.button("Ask"):
    if user_query:
        try:
            relevant_schemas = find_relevant_schemas(user_query)
            if relevant_schemas:
                description = relevant_schemas[0]["description"]
                columns = relevant_schemas[0]["columns"]

                prompt_string = generate_prompt(user_query, relevant_schemas)
                generated_sql = custom_prompt(prompt_string)
                sql_result = execute_sql_query(generated_sql)
                human_friendly_message = create_human_friendly_message(sql_result, user_query)

                add_to_memory(user_query, generated_sql, sql_result, human_friendly_message)

                # Display current conversation
                st.subheader("Current Conversation")
                st.write("**User Query:**", user_query)
                st.write("**Generated SQL:**", generated_sql)
                st.write("**SQL Result:**", sql_result)
                st.write("**Human-Friendly Message:**", human_friendly_message)
                st.markdown("---")

                # Display conversation history
                display_conversation_memory()

            else:
                st.error("No relevant schemas found for the query.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.button("Clear Memory"):
    clear_memory()
