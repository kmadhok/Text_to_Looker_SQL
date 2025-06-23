import streamlit as st
import pandas as pd
import os

def data_explorer_page():
    st.set_page_config(page_title="Data Explorer", page_icon="📊")
    st.title("📊 Data Explorer")
    st.markdown("Here are the first 100 rows of each CSV file found in the `data/` directory.")

    data_dir = "data"
    if not os.path.exists(data_dir):
        st.warning(f"The `{data_dir}` directory was not found.")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        st.info(f"No CSV files found in the `{data_dir}` directory.")
        return

    for filename in sorted(csv_files):
        st.subheader(f"📄 {filename}")
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath, nrows=100)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not read {filename}: {e}")

if __name__ == "__main__":
    data_explorer_page() 