import argparse
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- Pricing Constants ---
# Prices per 1 million tokens for gemini-2.5-flash
PRICE_INPUT_PER_MILLION_TOKENS = 0.030
PRICE_OUTPUT_PER_MILLION_TOKENS = 2.50

# --- Utility Functions ---

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

def clean_llm_response(text: str, response_type: str = "sql") -> str:
    """Cleans the raw text response from the LLM."""
    text = text.strip()
    if response_type == "json":
        if text.lower().startswith("```json"):
            text = text[7:].strip()
        elif text.lower().startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    elif response_type == "sql":
        if text.lower().startswith("```sql"):
            text = text[5:].strip()
        elif text.lower().startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text

# --- Gemini Generation Functions ---

def generate_sql(question: str, schema: str, api_key: str) -> tuple[str, int, int]:
    """Generates SQL query from a question and schema using Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

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
- For measures, use the `type` (e.g., `count`, `sum`, `average`) to construct the aggregation function, and the `SQL` part as its argument.
- Joins between explores are provided with the `ON` condition. The fields in `sql_on` use the format `explore_name.field_name`. You must map these to the correct table aliases in your `JOIN` clauses.
User Question: "{question}"
Based on the schema and guidelines, please generate a single Google BigQuery SQL query that answers the question.
SQL Query:
"""
    try:
        response = model.generate_content(prompt)
        sql_query = clean_llm_response(response.text, "sql")
        
        # Extract token usage
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        return sql_query, input_tokens, output_tokens
    except Exception as e:
        return f"An error occurred while generating SQL: {e}", 0, 0

def generate_write_query_params(question: str, schema: str, metadata: dict, api_key: str) -> tuple[dict, int, int]:
    """Generates Looker WriteQuery parameters from a question and schema using Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
You are an expert Looker developer. Your task is to generate the parameters for a Looker `WriteQuery` based on a user's question and the provided database schema.
Here is the database schema based on Looker explores:
--- SCHEMA START ---
{schema}
--- SCHEMA END ---
Guidelines:
- **model**: Use the model name "{metadata['model']}".
- **view**: The primary explore to use for the query.
- **fields**: A list of `explore_name.field_name` strings for all dimensions and measures needed.
- **filters**: A dictionary of `explore_name.field_name: value` for any filters.
- **sorts**: A list of sort strings, e.g., `['customers.count desc']`.
- **limit**: A string for the row limit. Use "1" for "the most/least" and a reasonable number like "10" for "top/bottom" queries.
User Question: "{question}"
Based on the schema and guidelines, generate a single, valid JSON object representing the parameters for the `WriteQuery`. Do not add any text before or after the JSON object.
Example for "Top 5 states by number of customers?":
```json
{{
    "model": "retail",
    "view": "customers",
    "fields": ["customers.state", "customers.count"],
    "filters": {{}},
    "sorts": ["customers.count desc"],
    "limit": "5"
}}
```
JSON Output for the user question:
"""
    try:
        response = model.generate_content(prompt)
        cleaned_response = clean_llm_response(response.text, "json")
        params = json.loads(cleaned_response)

        # Extract token usage
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count

        return params, input_tokens, output_tokens
    except (json.JSONDecodeError, Exception) as e:
        raw_response_text = "Could not get raw response text."
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            raw_response_text = e.response.text
        error_info = {
            "error": f"Failed to generate or parse WriteQuery params: {e}",
            "raw_response": raw_response_text
        }
        return error_info, 0, 0

# --- Reusable Function for Streamlit ---

def run_text_to_looker(question: str, metadata_file: str = "fake_metadata.json"):
    """
    Main function to be called by the Streamlit app.
    Takes a question and returns the WriteQuery params, SQL query, and token counts.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please create a .env file or set the environment variable.")

    metadata = load_metadata(metadata_file)
    schema_prompt = generate_schema_prompt(metadata)

    # Get WriteQuery params and their token usage
    write_query_params, wq_in_tokens, wq_out_tokens = generate_write_query_params(question, schema_prompt, metadata, api_key)
    
    # Get SQL query and its token usage
    sql_query, sql_in_tokens, sql_out_tokens = generate_sql(question, schema_prompt, api_key)

    total_input_tokens = wq_in_tokens + sql_in_tokens
    total_output_tokens = wq_out_tokens + sql_out_tokens

    return write_query_params, sql_query, total_input_tokens, total_output_tokens

# --- Main Execution for Command-Line Use ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Looker Query and SQL from a natural language question.")
    parser.add_argument("question", type=str, help="The natural language question to convert.")
    parser.add_argument("--metadata_file", type=str, default="fake_metadata.json", help="Path to the metadata JSON file.")
    args = parser.parse_args()

    try:
        write_query_params, sql_query, total_input, total_output = run_text_to_looker(args.question, args.metadata_file)

        input_cost = (total_input / 1_000_000) * PRICE_INPUT_PER_MILLION_TOKENS
        output_cost = (total_output / 1_000_000) * PRICE_OUTPUT_PER_MILLION_TOKENS
        total_cost = input_cost + output_cost

        print("\n--- User Question ---")
        print(args.question)
        print("\n--- Generated Looker WriteQuery Parameters ---")
        print(json.dumps(write_query_params, indent=2))
        print("\n--- Generated SQL Query ---")
        print(sql_query)
        print("\n--- Token Usage ---")
        print(f"Input Tokens: {total_input}")
        print(f"Output Tokens: {total_output}")
        print(f"Estimated Cost: ${total_cost:.6f}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")