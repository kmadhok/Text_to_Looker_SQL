import argparse
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

def load_metadata(filepath: str) -> dict:
    """Loads the metadata from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_schema_prompt(metadata: dict) -> str:
    """Generates a schema description from metadata for the LLM prompt."""
    prompt_parts = []
    for explore in metadata['explores']:
        explore_name = explore['name']
        table_name = explore['sql_table_name']
        prompt_parts.append(f"Explore: {explore_name} (Table: {table_name})")
        prompt_parts.append("  Fields:")

        dimensions = explore['fields']['dimensions']
        measures = explore['fields']['measures']

        for dim in dimensions:
            prompt_parts.append(f"    - {explore_name}.{dim['name']} (Dimension, Label: '{dim['label']}', Type: {dim['type']}, SQL: {dim['sql']})")

        for mea in measures:
            prompt_parts.append(f"    - {explore_name}.{mea['name']} (Measure, Label: '{mea['label']}', Type: {mea['type']}, SQL: {mea['sql']})")

        if 'joins' in explore and explore['joins']:
            prompt_parts.append("  Joins:")
            for join in explore['joins']:
                joined_explore = join['name']
                sql_on = join['sql_on'].replace(';;', '').strip()
                prompt_parts.append(f"    - Joins with '{joined_explore}' on: {sql_on}")
        prompt_parts.append("-" * 20)
    return "\n".join(prompt_parts)

def generate_sql(question: str, schema: str, api_key: str) -> str:
    """Generates SQL query from a question and schema using Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
You are an expert SQL developer. Your task is to write a Google BigQuery SQL query based on a user's question and the provided database schema.

Here is the database schema based on Looker explores:

--- SCHEMA START ---
{schema}
--- SCHEMA END ---

Guidelines:
- The schema describes 'explores', which are logical tables. Each explore has a `sql_table_name` which you should use in `FROM` and `JOIN` clauses. It's good practice to use table aliases.
- Fields are listed with their type (Dimension or Measure) and their SQL definition.
- When you use a field, you must use its `SQL` definition. Replace `${{TABLE}}` in the `SQL` definition with the alias of the table that contains the field.
- For measures, use the `type` (e.g., `count`, `sum`, `average`) to construct the aggregation function, and the `SQL` part as its argument. For example, a measure of type `sum` with `SQL: ${{TABLE}}.qty` should be translated to `SUM(table_alias.qty)`. A measure of type `count` with `SQL: *` should be `COUNT(*)`.
- Joins between explores are provided with the `ON` condition. The fields in `sql_on` use the format `explore_name.field_name`. You must map these to the correct table aliases in your `JOIN` clauses.

User Question: "{question}"

Based on the schema and guidelines, please generate a single Google BigQuery SQL query that answers the question.

SQL Query:
"""
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's just the SQL query
        sql_query = response.text.strip()
        if sql_query.lower().startswith("```sql"):
            sql_query = sql_query[5:].strip()
        if sql_query.lower().startswith("```"):
            sql_query = sql_query[3:].strip()
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3].strip()
        return sql_query
    except Exception as e:
        return f"An error occurred while generating SQL: {e}"

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please create a .env file with GOOGLE_API_KEY='your_key' or set the environment variable.")

    parser = argparse.ArgumentParser(description="Generate SQL from a natural language question using Gemini.")
    parser.add_argument("question", type=str, help="The natural language question to convert to SQL.")
    parser.add_argument("--metadata_file", type=str, default="fake_metadata.json", help="Path to the metadata JSON file.")

    args = parser.parse_args()

    metadata = load_metadata(args.metadata_file)
    schema_prompt = generate_schema_prompt(metadata)
    
    sql_query = generate_sql(args.question, schema_prompt, api_key)

    print("\n--- User Question ---")
    print(args.question)
    print("\n--- Generated SQL Query ---")
    print(sql_query) 