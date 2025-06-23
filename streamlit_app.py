#!/usr/bin/env python3
"""
Streamlit application for Looker Query Generation.
A web interface that converts natural language to Looker SDK WriteQuery format.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import sys
import os

# Add current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

# Import our existing functionality
try:
    from chat_gemini import GeminiRAGChat, WriteQuerySchema
    import fake_looker
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Looker Query Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chat_system():
    """Initialize the chat system once and cache it."""
    try:
        return GeminiRAGChat()
    except Exception as e:
        st.error(f"Failed to initialize chat system: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data from CSV files."""
    data = {}
    data_dir = Path("data")
    
    if not data_dir.exists():
        return None
    
    try:
        for csv_file in data_dir.glob("*.csv"):
            table_name = csv_file.stem
            df = pd.read_csv(csv_file)
            data[table_name] = df
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_data_summary(data):
    """Generate summary statistics for the data."""
    if not data:
        return None
    
    summary = {}
    total_rows = 0
    total_columns = 0
    
    for table_name, df in data.items():
        rows, cols = df.shape
        total_rows += rows
        total_columns += cols
        
        summary[table_name] = {
            "rows": rows,
            "columns": cols,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "column_types": df.dtypes.to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }
    
    summary["totals"] = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_tables": len(data)
    }
    
    return summary

def display_query_result(query_dict: Dict[str, Any], result: Optional[str], narrative: str):
    """Display the query and its results in a formatted way."""
    
    # Display the generated query
    st.subheader("🔍 Generated Looker WriteQuery")
    
    with st.container():
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        st.json(query_dict)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Copy to clipboard button
    query_json = json.dumps(query_dict, indent=2)
    if st.button("📋 Copy Query to Clipboard"):
        st.code(query_json, language="json")
        st.success("Query formatted for copying!")
    
    # Display results if available
    if result:
        try:
            result_data = json.loads(result)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Query Results")
                st.markdown(f'<div class="success-box">**Narrative**: {narrative}</div>', unsafe_allow_html=True)
                
                # Show results as dataframe if possible
                if result_data and isinstance(result_data, list):
                    df = pd.DataFrame(result_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv,
                        file_name="looker_query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.json(result_data)
            
            with col2:
                st.subheader("📈 Quick Stats")
                if result_data and isinstance(result_data, list):
                    st.metric("Rows Returned", len(result_data))
                    if result_data:
                        st.metric("Columns", len(result_data[0]))
                
        except json.JSONDecodeError:
            st.error("Could not parse query results")
            st.text(result)

def main_query_page():
    """Main page for query generation."""
    st.markdown('<h1 class="main-header">🔍 Looker Query Generator</h1>', unsafe_allow_html=True)
    
    # Initialize chat system
    chat_system = initialize_chat_system()
    
    if not chat_system:
        st.error("❌ Chat system initialization failed. Please check your setup.")
        return
    
    # API key status
    api_key_status = "✅ Connected" if chat_system.gemini_available else "⚠️ Limited (No Gemini API)"
    st.sidebar.markdown(f"**API Status**: {api_key_status}")
    
    # Main query interface
    st.subheader("💬 Natural Language Query")
    
    # Example queries
    example_queries = [
        "total sales last 7 days",
        "customers by state", 
        "top selling product categories",
        "sales by region",
        "average order value by month",
        "products with highest inventory"
    ]
    
    with st.expander("💡 Example Queries", expanded=False):
        st.write("Try these example queries:")
        for i, query in enumerate(example_queries):
            if st.button(f"📝 {query}", key=f"example_{i}"):
                st.session_state.user_query = query
    
    # Query input
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., Show me total sales for the last 30 days by state",
        value=st.session_state.get('user_query', ''),
        key='query_input'
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            custom_limit = st.number_input("Row Limit", min_value=1, max_value=5000, value=500)
        with col2:
            show_debug = st.checkbox("Show Debug Info", value=False)
    
    # Generate query button
    if st.button("🚀 Generate Looker Query", type="primary", disabled=not user_query.strip()):
        if not user_query.strip():
            st.warning("Please enter a query!")
            return
        
        with st.spinner("🔄 Processing your query..."):
            try:
                # Process the query
                response = chat_system.chat(user_query.strip())
                
                if response.get("error"):
                    st.markdown(f'<div class="error-box">❌ **Error**: {response["error"]}</div>', unsafe_allow_html=True)
                else:
                    # Apply custom limit if different from default
                    query_dict = response["query"]
                    if custom_limit != 500:
                        query_dict["limit"] = custom_limit
                    
                    # Display results
                    display_query_result(
                        query_dict=query_dict,
                        result=response.get("result"),
                        narrative=response.get("narrative", "Query executed successfully")
                    )
                    
                    # Debug information
                    if show_debug:
                        st.subheader("🐛 Debug Information")
                        with st.expander("Field Documentation Used"):
                            st.text(response.get("field_docs", "No field docs available"))
                
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ **Unexpected Error**: {str(e)}</div>', unsafe_allow_html=True)

def data_explorer_page():
    """Page for exploring the raw data."""
    st.markdown('<h1 class="main-header">📊 Data Explorer</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_sample_data()
    
    if not data:
        st.error("❌ No data found. Please run `python make_synthetic_data.py` first.")
        return
    
    # Data summary
    summary = get_data_summary(data)
    
    if summary:
        st.subheader("📈 Data Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tables", summary["totals"]["total_tables"])
        with col2:
            st.metric("Total Rows", f"{summary['totals']['total_rows']:,}")
        with col3:
            st.metric("Total Columns", summary["totals"]["total_columns"])
        with col4:
            memory_mb = sum(summary[table]["memory_usage"] for table in data.keys()) / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Table selection
    st.subheader("🗂️ Explore Tables")
    
    selected_table = st.selectbox(
        "Select a table to explore:",
        options=list(data.keys()),
        format_func=lambda x: f"{x} ({data[x].shape[0]:,} rows, {data[x].shape[1]} columns)"
    )
    
    if selected_table:
        df = data[selected_table]
        
        # Table info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📋 {selected_table.title()} Table")
            
            # Display options
            show_options = st.columns(3)
            with show_options[0]:
                show_head = st.number_input("Show first N rows", min_value=5, max_value=100, value=10)
            with show_options[1]:
                show_info = st.checkbox("Show column info", value=True)
            with show_options[2]:
                show_stats = st.checkbox("Show statistics", value=False)
            
            # Data display
            st.dataframe(df.head(show_head), use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"💾 Download {selected_table}.csv",
                data=csv,
                file_name=f"{selected_table}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("📊 Table Stats")
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", df.shape[1])
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Column info
            if show_info:
                st.subheader("📝 Columns")
                for col, dtype in df.dtypes.items():
                    null_count = df[col].isnull().sum()
                    null_pct = (null_count / len(df)) * 100
                    st.text(f"{col}: {dtype} ({null_pct:.1f}% null)")
        
        # Statistical summary
        if show_stats:
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.subheader("📊 Quick Visualizations")
            
            viz_col = st.selectbox("Select column to visualize:", numeric_cols)
            
            if viz_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(df, x=viz_col, title=f"Distribution of {viz_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(df, y=viz_col, title=f"Box Plot of {viz_col}")
                    st.plotly_chart(fig_box, use_container_width=True)

def schema_documentation_page():
    """Page for displaying schema and field documentation."""
    st.markdown('<h1 class="main-header">📚 Schema Documentation</h1>', unsafe_allow_html=True)
    
    # Load metadata
    try:
        with open("fake_metadata.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        st.error("❌ Metadata file not found. Please run `python make_synthetic_data.py` first.")
        return
    except json.JSONDecodeError:
        st.error("❌ Invalid metadata file format.")
        return
    
    # Model overview
    st.subheader(f"🏗️ Model: {metadata['model']}")
    st.write(f"This model contains **{len(metadata['explores'])}** explores representing different business entities.")
    
    # Explores overview
    explore_summary = []
    for explore in metadata['explores']:
        explore_summary.append({
            "Explore": explore['name'],
            "Label": explore['label'],
            "Table": explore['sql_table_name'],
            "Dimensions": len([f for f in explore['fields']['dimensions'] if not f.get('is_hidden', False)]),
            "Measures": len([f for f in explore['fields']['measures'] if not f.get('is_hidden', False)])
        })
    
    st.subheader("📊 Explores Summary")
    st.dataframe(pd.DataFrame(explore_summary), use_container_width=True)
    
    # Detailed explore documentation
    st.subheader("🔍 Detailed Field Documentation")
    
    selected_explore = st.selectbox(
        "Select an explore to view details:",
        options=[exp['name'] for exp in metadata['explores']],
        format_func=lambda x: f"{x.title()} ({next(exp['label'] for exp in metadata['explores'] if exp['name'] == x)})"
    )
    
    if selected_explore:
        explore_data = next(exp for exp in metadata['explores'] if exp['name'] == selected_explore)
        
        # Explore info
        st.markdown(f"**Table**: `{explore_data['sql_table_name']}`")
        st.markdown(f"**Description**: {explore_data['label']}")
        
        # Tabs for dimensions and measures
        dim_tab, meas_tab = st.tabs(["📏 Dimensions", "📊 Measures"])
        
        with dim_tab:
            dimensions = explore_data['fields']['dimensions']
            if dimensions:
                dim_data = []
                for dim in dimensions:
                    if not dim.get('is_hidden', False):
                        dim_data.append({
                            "Field": dim['name'],
                            "Label": dim['label'],
                            "Type": dim['type'],
                            "SQL": dim.get('sql', 'N/A')
                        })
                
                if dim_data:
                    st.dataframe(pd.DataFrame(dim_data), use_container_width=True)
                else:
                    st.info("No visible dimensions found.")
            else:
                st.info("No dimensions available.")
        
        with meas_tab:
            measures = explore_data['fields']['measures']
            if measures:
                meas_data = []
                for meas in measures:
                    if not meas.get('is_hidden', False):
                        meas_data.append({
                            "Field": meas['name'],
                            "Label": meas['label'], 
                            "Type": meas['type'],
                            "SQL": meas.get('sql', 'N/A')
                        })
                
                if meas_data:
                    st.dataframe(pd.DataFrame(meas_data), use_container_width=True)
                else:
                    st.info("No visible measures found.")
            else:
                st.info("No measures available.")
    
    # Field documentation files
    st.subheader("📄 Field Documentation Files")
    
    docs_dir = Path("lookml_docs")
    if docs_dir.exists():
        doc_files = list(docs_dir.glob("*.txt"))
        
        if doc_files:
            selected_doc = st.selectbox(
                "View field documentation:",
                options=[f.stem for f in doc_files],
                format_func=lambda x: x.replace('.', ' → ')
            )
            
            if selected_doc:
                doc_path = docs_dir / f"{selected_doc}.txt"
                try:
                    content = doc_path.read_text()
                    st.markdown(f"**Field**: `{selected_doc}`")
                    st.markdown(f'<div class="query-box">{content}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading documentation: {e}")
        else:
            st.info("No field documentation files found.")
    else:
        st.info("Documentation directory not found.")

def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("🔍 Looker Query Generator")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🚀 Query Generator", "📊 Data Explorer", "📚 Schema Documentation"],
        index=0
    )
    
    # Page routing
    if page == "🚀 Query Generator":
        main_query_page()
    elif page == "📊 Data Explorer":
        data_explorer_page()
    elif page == "📚 Schema Documentation":
        schema_documentation_page()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 About")
    st.sidebar.markdown("""
    This app converts natural language queries into Looker SDK WriteQuery format using:
    - 🤖 Google Gemini for query generation
    - 🔍 Vector search for field retrieval  
    - ✅ Pydantic validation for query structure
    - 🧪 Local fake Looker SDK for testing
    """)
    
    # System status
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check if required files exist
    files_status = {
        "🗂️ Data": Path("data").exists(),
        "📊 Metadata": Path("fake_metadata.json").exists(),
        "📚 Documentation": Path("lookml_docs").exists(),
        "🔍 Index": Path("faiss_index").exists()
    }
    
    for name, status in files_status.items():
        icon = "✅" if status else "❌"
        st.sidebar.markdown(f"{icon} {name}")

if __name__ == "__main__":
    main() 