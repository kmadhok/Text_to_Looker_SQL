# 🔍 Looker Query Generator - Streamlit App

A beautiful web interface for converting natural language queries into Looker SDK WriteQuery format.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Data is Generated**
   ```bash
   python make_synthetic_data.py
   python build_index.py
   ```

3. **Launch the App**
   ```bash
   python run_streamlit.py
   # OR
   streamlit run streamlit_app.py
   ```

4. **Open Browser**
   - Navigate to `http://localhost:8501`
   - Start querying!

## 📱 App Features

### 🚀 Query Generator (Main Page)
- **Natural Language Input**: Type questions like "total sales last 7 days"
- **WriteQuery Generation**: See the exact Looker SDK query format
- **Live Results**: Execute queries and see results immediately
- **Copy to Clipboard**: Easy copying of generated queries
- **Example Queries**: Pre-built examples to get started

### 📊 Data Explorer
- **Table Browser**: Explore all synthetic data tables
- **Interactive Filtering**: View subsets of data
- **Statistics**: Column info, null counts, data types
- **Visualizations**: Histograms and box plots for numeric data
- **Export Options**: Download data as CSV

### 📚 Schema Documentation  
- **Model Overview**: Complete schema structure
- **Field Definitions**: Detailed field documentation
- **SQL Mappings**: See how fields map to SQL
- **Dimensions vs Measures**: Organized field categories

## 🎯 Main Focus: Text-to-Query

The primary feature is the **text-to-query generation**:

```python
# Input
"Show me total sales for customers in California last month"

# Output (Looker WriteQuery)
{
  "model": "retail",
  "explore": "order_items",
  "fields": ["order_items.total_sales", "customers.state"],
  "filters": {
    "customers.state": "CA",
    "orders.order_date": "1 month"
  },
  "limit": 500
}
```

## 🔧 Configuration

- **Google API Key**: Set `GOOGLE_API_KEY` environment variable for full functionality
- **Local Mode**: Works without API keys using keyword-based search
- **Custom Limits**: Adjust row limits in the advanced options
- **Debug Mode**: Enable to see field retrieval details

## 📁 File Dependencies

The app requires these files to be present:
- `fake_looker.py` - Local Looker SDK implementation
- `chat_gemini.py` - RAG chat functionality  
- `data/` - Synthetic data CSV files
- `fake_metadata.json` - Looker metadata structure
- `lookml_docs/` - Field documentation
- `faiss_index/` - Vector search index

## 🛠️ Troubleshooting

**App won't start?**
- Check that all dependencies are installed
- Ensure you're in the project root directory
- Run `python make_synthetic_data.py` first

**No API key warnings?**  
- The app works without API keys using local search
- Set `GOOGLE_API_KEY` for full Gemini functionality

**Empty data?**
- Run `python make_synthetic_data.py` to generate data
- Check that `data/` directory contains CSV files

## 🎨 Customization

The app uses custom CSS for styling. You can modify the appearance by editing the styles in `streamlit_app.py`.

**Color Scheme**:
- Primary: Blue (#1f77b4)
- Success: Green (#28a745) 
- Error: Red (#dc3545)
- Background: Light gray (#f8f9fa) 