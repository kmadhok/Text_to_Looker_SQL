#!/usr/bin/env python3
"""
Demo script showcasing the complete Looker sandbox implementation.
Demonstrates all three components: data generation, fake SDK, and RAG chat (with/without Gemini).
"""

import json
import fake_looker
from pathlib import Path

def demo_data_generation():
    """Demo the synthetic data generation."""
    print("🚀 DEMO: Data Generation")
    print("=" * 40)
    
    # Check if data exists
    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.glob("*.csv"))
        print(f"✅ Found {len(files)} CSV files:")
        for file in files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   📄 {file.name}: {size_mb:.1f} MB")
        
        # Show sample data
        import pandas as pd
        customers = pd.read_csv("data/customers.csv")
        print(f"\n📊 Sample customer data (first 3 rows):")
        print(customers.head(3).to_string())
    else:
        print("❌ No data found. Run: python make_synthetic_data.py")

def demo_fake_sdk():
    """Demo the fake Looker SDK capabilities."""
    print("\n\n🔧 DEMO: Fake Looker SDK")
    print("=" * 40)
    
    # Demo 1: List models and explores
    print("📋 Available models:")
    models = fake_looker.all_lookml_models()
    for model in models:
        print(f"   - {model['name']}")
    
    model_info = fake_looker.lookml_model("retail")
    print(f"\n📊 Explores in 'retail' model:")
    for explore in model_info["explores"]:
        print(f"   - {explore['name']}")
    
    # Demo 2: Explore fields
    explore = fake_looker.lookml_model_explore("retail", "order_items")
    print(f"\n📈 Fields in 'order_items' explore:")
    print(f"   Dimensions: {len(explore['fields']['dimensions'])}")
    print(f"   Measures: {len(explore['fields']['measures'])}")
    
    # Demo 3: Execute queries
    print(f"\n🔍 Sample queries:")
    
    queries = [
        {
            "name": "Total Sales",
            "query": {
                "model": "retail",
                "explore": "order_items",
                "fields": ["total_sales"],
                "limit": 1
            }
        },
        {
            "name": "Top 3 States by Customer Count",
            "query": {
                "model": "retail", 
                "explore": "customers",
                "fields": ["state", "count"],
                "sorts": ["count desc"],
                "limit": 3
            }
        },
        {
            "name": "Sales by Category",
            "query": {
                "model": "retail",
                "explore": "order_items",
                "fields": ["products.category", "total_sales"],
                "sorts": ["total_sales desc"],
                "limit": 5
            }
        }
    ]
    
    for demo in queries:
        print(f"\n   📊 {demo['name']}:")
        try:
            result = fake_looker.run_inline_query("json", demo["query"])
            data = json.loads(result)
            if data:
                print(f"      Result: {data[0]}")
            else:
                print("      No data returned")
        except Exception as e:
            print(f"      ❌ Error: {e}")

def demo_field_docs():
    """Demo the field documentation and indexing."""
    print("\n\n📚 DEMO: Field Documentation & Indexing")
    print("=" * 50)
    
    docs_dir = Path("lookml_docs")
    if docs_dir.exists():
        docs = list(docs_dir.glob("*.txt"))
        print(f"✅ Found {len(docs)} field documentation files:")
        for doc in docs[:3]:  # Show first 3
            field_name = doc.stem
            content = doc.read_text()[:100] + "..."
            print(f"   📄 {field_name}: {content}")
        
        if len(docs) > 3:
            print(f"   ... and {len(docs) - 3} more")
    else:
        print("❌ No documentation found")
    
    # Check if index exists
    index_dir = Path("faiss_index")
    if index_dir.exists():
        print(f"\n✅ FAISS index built and ready")
        try:
            import pickle
            with open("faiss_index/doc_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            print(f"   📊 Indexed {len(metadata)} fields")
        except:
            print("   ⚠️  Could not read index metadata")
    else:
        print("❌ No index found. Run: python build_index.py")

def demo_chat_interface():
    """Demo the chat interface (with/without Gemini)."""
    print("\n\n💬 DEMO: Chat Interface")
    print("=" * 30)
    
    try:
        from chat_gemini import GeminiRAGChat
        
        chat = GeminiRAGChat()
        
        print("🔍 Testing field retrieval (works without API key):")
        test_queries = [
            "total sales revenue",
            "customer demographics age",
            "product categories"
        ]
        
        for query in test_queries:
            fields = chat.retrieve_relevant_fields(query)
            if fields and "No relevant fields found" not in fields:
                # Extract first field name
                first_line = fields.split('\n')[0]
                print(f"   '{query}' → {first_line}")
            else:
                print(f"   '{query}' → No matches")
        
        print(f"\n🤖 Gemini Integration Status:")
        if chat.gemini_available:
            print("   ✅ Gemini API available - Full functionality enabled")
        else:
            print("   ⚠️  Gemini API not available (GOOGLE_API_KEY not set)")
            print("   💡 Set GOOGLE_API_KEY to enable query generation")
            
    except Exception as e:
        print(f"❌ Chat interface error: {e}")

def demo_end_to_end():
    """Demo a complete end-to-end flow."""
    print("\n\n🎯 DEMO: End-to-End Flow")
    print("=" * 35)
    
    print("🔄 Simulating: 'Show me total sales last 30 days'")
    print("\n📋 Steps that would happen with full API access:")
    print("   1. 🔍 Retrieve relevant fields from index")
    print("   2. 🤖 Generate WriteQuery with Gemini")
    print("   3. ✅ Validate query with Pydantic")
    print("   4. 🚀 Execute via fake Looker SDK") 
    print("   5. 📊 Return formatted results")
    
    print(f"\n💡 Manual equivalent query:")
    manual_query = {
        "model": "retail",
        "explore": "order_items", 
        "fields": ["total_sales"],
        "filters": {"orders.order_date": "30 days"},
        "limit": 1
    }
    print(f"   {json.dumps(manual_query, indent=4)}")
    
    try:
        result = fake_looker.run_inline_query("json", manual_query)
        data = json.loads(result)
        if data:
            sales = data[0]["total_sales"]
            print(f"\n📈 Result: Total sales (last 30 days): ${sales:,.2f}")
        else:
            print(f"\n📈 Result: No data found")
    except Exception as e:
        print(f"\n❌ Query failed: {e}")

def main():
    """Run complete demo."""
    print("🚀 LOOKER SANDBOX DEMO")
    print("=" * 60)
    print("A three-stage sandbox that mimics Looker behavior:")
    print("1. 📊 Synthetic data generation")
    print("2. 🔧 Fake Looker SDK") 
    print("3. 💬 Gemini RAG chat interface")
    print()
    
    demo_data_generation()
    demo_fake_sdk()
    demo_field_docs()
    demo_chat_interface()
    demo_end_to_end()
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 20)
    print("🏁 Next Steps:")
    print("   1. Set GOOGLE_API_KEY environment variable or create .env file")
    print("      Example .env file content: GOOGLE_API_KEY=your-api-key-here")
    print("   2. Run: python chat_gemini.py for interactive chat")
    print("   3. Replace fake_looker with real looker_sdk when ready")
    print("\n💡 All components ready for production transition!")
    print("🔑 .env file support available for easy API key management")

if __name__ == "__main__":
    main() 