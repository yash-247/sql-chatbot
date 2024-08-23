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

os.environ['OPENAI_API_KEY'] = 'KEY HERE' #API KEY 

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

# Few-shot examples
few_shot_examples = [
    {
        "question": "Get the total number of sessions and pageviews for each user in the last week.",
        "sql": "SELECT user_id, COUNT(*) AS sessions, SUM(pageviews) AS total_pageviews FROM website_sessions JOIN website_pageviews USING (website_session_id) WHERE created_at >= NOW() - INTERVAL 7 DAY GROUP BY user_id;"
    },
    {
        "question": "Find the top 5 products by sales in the last month.",
        "sql": "SELECT product_id, SUM(price_usd) AS total_sales FROM order_items WHERE created_at >= NOW() - INTERVAL 1 MONTH GROUP BY product_id ORDER BY total_sales DESC LIMIT 5;"
    },
    {
        "question": "Get the total revenue and cost for each day in the last month.",
        "sql": "SELECT DATE(created_at) AS date, SUM(price_usd) AS total_revenue, SUM(cogs_usd) AS total_cost FROM orders WHERE created_at >= NOW() - INTERVAL 1 MONTH GROUP BY date;"
    }
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

# Function to find the most relevant example
def find_most_relevant_example(user_query, examples):
    user_embedding = generate_embeddings(user_query)
    example_embeddings = [generate_embeddings(ex['question']) for ex in examples]
    
    similarities = [np.dot(user_embedding, ex_emb) for ex_emb in example_embeddings]
    most_relevant_index = np.argmax(similarities)
    
    return examples[most_relevant_index]

# Update the prompt template
template = """
You are an SQL expert. Based on the table schemas below and the provided example, write a MySQL query that would answer the user's question. You will not generate new columns and tables other than the ones provided in the schema.

Schemas:
{description}

Columns:
{columns}

Example: Use this example as reference if necessary else generate your own SQL query based on the question provided.
{examples}

Question: {question}

SQL Query:
"""

# Function to generate the prompt
def generate_prompt(question, relevant_schemas, relevant_example):
    schema_descriptions = []
    schema_columns = []
    
    for schema in relevant_schemas:
        schema_descriptions.append(schema["description"])
        schema_columns.append(", ".join(schema["columns"]))
    
    schema_descriptions_str = "\n".join(schema_descriptions)
    schema_columns_str = "\n".join(schema_columns)

    # Use only the most relevant example
    example_str = f"Question: {relevant_example['question']}\nSQL Query: {relevant_example['sql']}"
    
    prompt = template.format(description=schema_descriptions_str, columns=schema_columns_str, examples=example_str, question=question)
    print("\nGenerated Prompt:")
    print(prompt)
    return prompt

prompt = ChatPromptTemplate.from_template(template)

db_uri = "mysql+mysqlconnector://root:12345678@localhost:3306/mavenfuzzyfactory"
db = SQLDatabase.from_uri(db_uri)

def get_schema(_):
    return db.get_table_info()

# Initialize the Google Gemini LLM with the API key
gemini_api_key = "AIzaSyDj9Ou99iNvCkqnongLtJNKCUAfh1leIc8"
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key, temperature=0, convert_system_message_to_human=True)

# Define a custom prompt function
def custom_prompt(prompt_string):
    print("\nPrompt being sent to Google Gemini API:")
    print(prompt_string)
    messages = [HumanMessage(content=prompt_string)]
    result = llm(messages)
    clean_sql = result.content.strip().strip('```sql').strip('```')
    print("\nGenerated SQL:")
    print(clean_sql)
    return clean_sql

# Execute the SQL query and return the results
def execute_sql_query(sql_query):
    result = db.run(sql_query)
    return result

# Define a function to create a human-readable message from the SQL result
def create_human_friendly_message(sql_result, question):
    prompt_string = f"Based on the SQL query result below, provide a human-friendly explanation for the following question. Do not create your own tables and answers if SQL Query is null. Display a friendly message and do not create anything of your own:\n\nQuestion: {question}\n\nSQL Query Result: {sql_result}"
    print("\nPrompt being sent to Google Gemini API for explanation:")
    print(prompt_string)
    messages = [HumanMessage(content=prompt_string)]
    result = llm(messages)
    print("\nHuman-friendly explanation:")
    print(result.content)
    return result.content

def generate_and_send_prompt(inputs):
    prompt_string = inputs.to_string()
    matches = re.search(r'Question: (.+)', prompt_string)
    if matches:
        question = matches.group(1)
        relevant_schemas = find_relevant_schemas(question)
        print("\nRelevant Schemas:")
        print(relevant_schemas)
        relevant_example = find_most_relevant_example(question, few_shot_examples)
        print("\nMost Relevant Example:")
        print(relevant_example)
        prompt_string = generate_prompt(question, relevant_schemas, relevant_example)
        generated_sql = custom_prompt(prompt_string)
        sql_result = execute_sql_query(generated_sql)
        print("\nSQL Result:")
        print(sql_result)
        human_friendly_message = create_human_friendly_message(sql_result, question)
        return human_friendly_message
    else:
        raise ValueError("Could not extract the question from the prompt string.")

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | generate_and_send_prompt
    | StrOutputParser()
)

# User query here
user_query = "Get the total number of sessions and pageviews for each user from july 19th 2012 till july 21st 2012."
print("\nUser Query:")
print(user_query)

relevant_schemas = find_relevant_schemas(user_query)
description = relevant_schemas[0]["description"]
columns = relevant_schemas[0]["columns"]

# Find the most relevant example
relevant_example = find_most_relevant_example(user_query, few_shot_examples)
example_str = f"Question: {relevant_example['question']}\nSQL Query: {relevant_example['sql']}"

print("\nMost Relevant Example:")
print(example_str)

# Update the order of keys to match the prompt template
result = sql_chain.invoke({
    "question": user_query, 
    "description": description, 
    "columns": columns,
    "examples": example_str
})
print("\nFinal Result:")
print(result)