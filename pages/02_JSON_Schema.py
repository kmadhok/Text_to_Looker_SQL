import streamlit as st
import json
import os

def json_schema_page():
    st.set_page_config(page_title="JSON Schema", page_icon="📜")
    st.title("📜 JSON Schema (`fake_metadata.json`)")
    st.markdown("This is the JSON metadata that the AI uses to understand the database schema, including explores, fields, and joins.")

    metadata_file = "fake_metadata.json"

    if not os.path.exists(metadata_file):
        st.error(f"The file `{metadata_file}` was not found.")
        st.info("Please make sure the metadata file is in the root directory of the application.")
        return

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        st.json(metadata)
    except json.JSONDecodeError:
        st.error(f"Could not parse the JSON from `{metadata_file}`. Please ensure it is a valid JSON file.")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    json_schema_page() 