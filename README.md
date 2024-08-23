# Enhancing Database Interactions Using Conversational Chatbot

## Overview

This project focuses on the development of a conversational AI chatbot designed to facilitate natural language querying of databases. The chatbot leverages advanced language models and various technologies to streamline and enhance database interactions, making it easier for users to access and analyze data without needing extensive SQL knowledge.

## Table of Contents

### Project Overview

### Technologies Used

### Architecture

### Installation

### Usage

### Features

### Contributing

### License

## Technologies Used

LLM: Google Gemini LLM
Vector Store: FAISS (Facebook AI Similarity Search)
Embeddings: SentenceTransformers
Conversational AI Framework: LangChain
Database: MySQL, Graph Database
Data Visualization: Integrated data visualization features using appropriate libraries and tools

## Architecture

The architecture of the chatbot is designed to convert natural language queries into SQL, execute them against the MySQL database, and return results in a user-friendly format. Here's a high-level overview:

### Query Parsing: The chatbot uses LangChain to parse the user's natural language input.

### Embedding Generation: SentenceTransformers generate embeddings of both the input query and database schema.

### Schema Matching: FAISS is utilized to match the query against the database schema and generate the most relevant SQL.

### SQL Execution: The chatbot executes the generated SQL on the MySQL database.

### Results Presentation: Results are presented to the user in a human-friendly format, with options for data visualization.

![alt text](<Vector Searching.jpg>)

# Approach 2
# USING FEW SHOTS LEARNING
![alt text](<Vector Searching with Few shots learning.jpg>)

# Approach 3
# Using knowledge graph as database
![alt text](<Knowledge Graph.jpg>)

# Database description
![alt text](<mavenfuzzyfactory DBSchema.png>)

Installation
To set up this project locally:

Configure your API keys and database credentials in the config.py file.
Configure the SQL Databases accordingly.
Usage
Run the chatbot:open in terminal and run the code
streamlit run sql-chatbot.py

Input natural language queries and receive SQL-generated results, along with visualizations if needed.

# Features

#### Natural Language Querying: Simplify database interactions using conversational AI.

#### SQL Generation: Automatically convert natural language into accurate SQL queries.

#### Data Visualization: View results in graphs and charts.

#### Extensible Architecture: Easily modify and extend the chatbot’s capabilities.


# Results
![alt text](<Screenshot 2024-06-25 220514.png>)

![alt text](<Screenshot 2024-06-25 220743.png>)

![alt text](<Screenshot 2024-06-29 015659.png>)

![alt text](<Screenshot 2024-06-29 020300.png>)
![alt text](<Screenshot 2024-07-19 002158.png>) 
![alt text](<Screenshot 2024-07-19 001311.png>)
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch: git checkout -b feature-branch-name.
Make your changes.
Commit and push: git push origin feature-branch-name.
Create a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any inquiries or issues, feel free to contact:
mryashkaushal@gmail.com
