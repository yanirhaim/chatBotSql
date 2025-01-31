import os
import openai
import psycopg2
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import sqlparse
import dotenv

dotenv.load_dotenv()
# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Establish a database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
        )
        return conn
    except psycopg2.Error as e:
        print("Failed to connect to the database:", e)
        return None


# Retrieve the database schema
def get_db_schema():
    conn = get_db_connection()
    if not conn:
        return {}
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    schema = {}
    for table_name, column_name in rows:
        if table_name not in schema:
            schema[table_name] = []
        schema[table_name].append(column_name)
    return schema


# Convert the schema dictionary to a string
def schema_to_string(schema):
    schema_str = ""
    for table, columns in schema.items():
        schema_str += f"{table} ({', '.join(columns)})\n"
    return schema_str


# Initialize the OpenAI model
llm = OpenAI(temperature=0.5, max_tokens=1000)

# Define a prompt template for generating SQL queries
prompt_template = PromptTemplate(
    input_variables=["user_prompt", "schema"],
    template="""
    You are a smart assistant. Based on the user's request: {user_prompt},
    generate a valid SQL query that matches the given database schema, make sure that the properties are available.
    Ensure that all table and column names are correct.

    Schema:
    {schema}

    SQL Query:
    """
)


def generate_query(user_prompt, schema_str):
    prompt = prompt_template.format(user_prompt=user_prompt, schema=schema_str)
    response = llm(prompt)
    query = response.strip()
    return query


def validate_query(query, schema):
    # Use sqlparse to validate the structure of the query
    parsed_query = sqlparse.parse(query)
    if not parsed_query:
        return False

    # Basic check: Ensure the query contains only valid table and column names
    for statement in parsed_query:
        tokens = statement.tokens
        for token in tokens:
            if token.ttype is sqlparse.tokens.Name:
                name = token.value
                if name not in schema and not any(name in cols for cols in schema.values()):
                    print(f"Invalid name in query: {name}")
                    return False
    return True


def execute_query(query, schema):
    if not validate_query(query, schema):
        print("Invalid query")
        return []

    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
    except psycopg2.Error as e:
        print("Error executing query:", e)
        result = []
    finally:
        cursor.close()
        conn.close()
    return result


def format_result(result):
    if not result:
        return "No results found."
    # Format the result as needed
    formatted_result = "\n".join([str(row) for row in result])
    return formatted_result


# Define a prompt template for generating sales pitch explanations
sales_pitch_prompt_template = PromptTemplate(
    input_variables=["query", "result"],
    template="""
    Based on the SQL query:
    {query}

    The result is:
    {result}

    You are a real estate agent. Explain the listings in a detailed and persuasive way, highlighting key features that would attract potential buyers. 
    You do not need to talk about the IDS of the properties, just give the user description of the properties.
    """
)


def generate_sales_pitch(query, result):
    formatted_result = format_result(result)
    sales_pitch_prompt = sales_pitch_prompt_template.format(query=query, result=formatted_result)
    response = llm(sales_pitch_prompt)
    sales_pitch = response.strip()
    return sales_pitch


def chatbot(user_prompt):
    # Retrieve the schema from the database
    schema = get_db_schema()
    schema_str = schema_to_string(schema)
    # Generate the SQL query based on the user prompt and schema
    query = generate_query(user_prompt, schema_str)
    # Execute the query
    result = execute_query(query, schema)
    # Generate a sales pitch explanation based on the query and result
    sales_pitch = generate_sales_pitch(query, result)
    return sales_pitch


# Example usage
user_prompt = "I am looking for a house that has 3 bedrooms"
response = chatbot(user_prompt)
print(response)