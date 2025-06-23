#!/usr/bin/env python3
"""
Gemini RAG Chat Interface for LookML Queries.
Retrieves field docs from FAISS index, generates Looker queries with Gemini, validates with Pydantic.
"""

import os
import json
import pickle
from typing import List, Optional, Dict, Any
from pathlib import Path
import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator
from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv
import fake_looker

# Load environment variables from .env file if it exists
load_dotenv()

# Pydantic schema for WriteQuery validation (mirrors Looker API)
class WriteQuerySchema(BaseModel):
    """Pydantic model for Looker WriteQuery API validation."""
    
    model: str = Field(description="LookML model name (e.g., 'retail')")
    explore: str = Field(description="Explore name (e.g., 'customers', 'orders')")
    fields: List[str] = Field(description="List of field names to select")
    filters: Optional[Dict[str, str]] = Field(default=None, description="Filter conditions")
    sorts: Optional[List[str]] = Field(default=None, description="Sort specifications")
    limit: Optional[int] = Field(default=500, description="Row limit")
    pivots: Optional[List[str]] = Field(default=None, description="Pivot fields (not supported)")
    total: Optional[bool] = Field(default=None, description="Include totals (not supported)")
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v != 'retail':
            raise ValueError(f"Invalid model '{v}'. Only 'retail' is supported.")
        return v
    
    @field_validator('explore')
    @classmethod
    def validate_explore(cls, v):
        valid_explores = ['customers', 'products', 'stores', 'orders', 'order_items']
        if v not in valid_explores:
            raise ValueError(f"Invalid explore '{v}'. Must be one of: {valid_explores}")
        return v
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        if v is not None and (v < 1 or v > 5000):
            raise ValueError("Limit must be between 1 and 5000")
        return v
    
    @field_validator('pivots')
    @classmethod
    def validate_no_pivots(cls, v):
        if v:
            raise ValueError("Pivots are not supported in this implementation")
        return v
    
    @field_validator('total')
    @classmethod
    def validate_no_totals(cls, v):
        if v:
            raise ValueError("Totals are not supported in this implementation")
        return v

class GeminiRAGChat:
    """Main chat interface with RAG retrieval and query generation."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.setup_gemini()
        self.load_index()
        self.load_field_metadata()
    
    def setup_gemini(self):
        """Setup Google Gemini API."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("⚠️  GOOGLE_API_KEY environment variable not set.")
            print("📝 You can set it with: export GOOGLE_API_KEY='your-api-key'")
            print("📝 Or create a .env file with: GOOGLE_API_KEY=your-api-key")
            print("🔑 Get an API key from: https://aistudio.google.com/app/apikey")
            self.gemini_available = False
            return
        
        genai.configure(api_key=api_key)
        
        # Set model (can be overridden with GEMINI_MODEL_ID env var)
        model_id = os.getenv('GEMINI_MODEL_ID', 'gemini-2.5-flash')
        self.model = genai.GenerativeModel(model_id)
        self.gemini_available = True
        print(f"🤖 Using Gemini model: {model_id}")
        print("✅ Google API key loaded successfully")
    
    def load_index(self):
        """Load the FAISS index for field retrieval."""
        try:
            if Path("faiss_index").exists():
                # Determine which embedding model was used to build the index
                api_key = os.getenv('GOOGLE_API_KEY')
                
                if api_key:
                    # Use Google Gemini embeddings (matching build process)
                    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
                    embed_model = GoogleGenAIEmbedding(
                        model_name="models/embedding-001",
                        api_key=api_key
                    )
                    print("🔗 Loading index with Google Gemini embeddings...")
                else:
                    # Use local embeddings (matching build process)
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    embed_model = HuggingFaceEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        trust_remote_code=True,
                        device="cpu"
                    )
                    print("🤖 Loading index with local sentence transformers...")
                
                storage_context = StorageContext.from_defaults(persist_dir="./faiss_index")
                self.index = load_index_from_storage(storage_context, embed_model=embed_model)
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=4,
                    response_mode="no_text"
                )
                print("📚 FAISS index loaded successfully")
            else:
                print("⚠️  FAISS index not found. Run build_index.py first.")
                self.index = None
                self.query_engine = None
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            # Fallback: try loading without specifying embedding model
            try:
                print("🔄 Trying to load index without embedding model...")
                storage_context = StorageContext.from_defaults(persist_dir="./faiss_index")
                self.index = load_index_from_storage(storage_context)
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=4,
                    response_mode="no_text"
                )
                print("📚 Index loaded with fallback method")
            except Exception as e2:
                print(f"❌ Fallback loading also failed: {e2}")
                self.index = None
                self.query_engine = None
    
    def load_field_metadata(self):
        """Load field metadata for fallback retrieval."""
        try:
            with open("faiss_index/doc_metadata.pkl", "rb") as f:
                self.field_metadata = pickle.load(f)
            print(f"📋 Loaded metadata for {len(self.field_metadata)} fields")
        except Exception as e:
            print(f"⚠️  Could not load field metadata: {e}")
            self.field_metadata = []
    
    def retrieve_relevant_fields(self, user_query: str) -> str:
        """Retrieve relevant field documentation based on user query."""
        # Try vector-based search first
        if self.query_engine:
            try:
                response = self.query_engine.query(user_query)
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    field_docs = []
                    for node in response.source_nodes[:4]:
                        field_name = node.metadata.get('field_name', 'Unknown')
                        content = node.text
                        field_docs.append(f"Field: {field_name}\n{content}\n")
                    return "\n".join(field_docs)
            except Exception as e:
                print(f"⚠️  Vector search failed: {e}")
        
        # Try keyword-based document store search
        keyword_docs_path = Path("faiss_index/keyword_docs.json")
        if keyword_docs_path.exists():
            try:
                print("🔍 Using keyword-based document search...")
                with open(keyword_docs_path, 'r') as f:
                    doc_store = json.load(f)
                
                keywords = user_query.lower().split()
                relevant_fields = []
                
                for doc_id, doc_data in doc_store.items():
                    content = doc_data['content'].lower()
                    field_name = doc_data['metadata'].get('field_name', 'Unknown')
                    
                    # Score based on keyword matches
                    score = sum(1 for keyword in keywords if keyword in content)
                    if score > 0:
                        relevant_fields.append((score, field_name, doc_data['content']))
                
                # Sort by score and return top results
                relevant_fields.sort(key=lambda x: x[0], reverse=True)
                field_docs = []
                for score, field_name, content in relevant_fields[:4]:
                    field_docs.append(f"Field: {field_name}\n{content}\n")
                
                if field_docs:
                    return "\n".join(field_docs)
                    
            except Exception as e:
                print(f"⚠️  Keyword search failed: {e}")
        
        # Fallback: simple keyword matching on loaded metadata
        if self.field_metadata:
            keywords = user_query.lower().split()
            relevant_fields = []
            
            for field_meta in self.field_metadata:
                field_content = field_meta['content'].lower()
                if any(keyword in field_content for keyword in keywords):
                    relevant_fields.append(f"Field: {field_meta['field_name']}\n{field_meta['content']}\n")
            
            if relevant_fields:
                return "\n".join(relevant_fields[:4])
        
        return "No relevant fields found. Available search methods failed."
    
    def generate_query(self, user_query: str, field_docs: str) -> Optional[Dict[str, Any]]:
        """Generate a Looker query using Gemini."""
        if not self.gemini_available:
            print("❌ Gemini API not available. Cannot generate queries.")
            return None
        
        prompt = f"""
You are an expert at writing Looker queries. Based on the user's request and the available field documentation, generate a valid WriteQuery JSON.

User Request: {user_query}

Available Fields:
{field_docs}

Requirements:
1. Return ONLY valid JSON in the WriteQuery format
2. Use only fields from the documentation above
3. Model must be "retail"
4. Explore must be one of: customers, products, stores, orders, order_items
5. If requested fields span multiple tables, DEFAULT to explore "orders". You may choose a different explore only when *all* requested fields reside in that single table.
6. Include appropriate filters if the user mentions specific criteria
7. If the user asks for the 'top', 'best', or 'most' of something, you MUST include a dimension for that 'something' and a measure to rank it. Also add a descending sort on the measure. For example, for 'top selling products', include a product dimension (like `products.name`), a sales measure (like `order_items.total_sales`), and a sort for `order_items.total_sales desc`.
8. Set reasonable limits (default 500, max 5000)
9. Do not use pivots or totals (not supported)

Example format:
{{
    "model": "retail",
    "explore": "orders",
    "fields": ["order_items.total_sales", "stores.region"],
    "filters": {{"orders.order_date": "30 days"}},
    "sorts": ["order_items.total_sales desc"],
    "limit": 100
}}

WriteQuery JSON:
"""

        try:
            response = self.model.generate_content(prompt)
            query_text = response.text.strip()
            
            # Extract JSON from response (remove any markdown formatting)
            if "```json" in query_text:
                query_text = query_text.split("```json")[1].split("```")[0].strip()
            elif "```" in query_text:
                query_text = query_text.split("```")[1].split("```")[0].strip()
            
            # Parse and validate JSON
            query_dict = json.loads(query_text)
            return query_dict
            
        except Exception as e:
            print(f"❌ Error generating query: {e}")
            return None
    
    def validate_query(self, query_dict: Dict[str, Any]) -> Optional[WriteQuerySchema]:
        """Validate query using Pydantic schema."""
        try:
            validated_query = WriteQuerySchema(**query_dict)
            return validated_query
        except Exception as e:
            print(f"❌ Query validation failed: {e}")
            return None
    
    def execute_query(self, query: WriteQuerySchema) -> Optional[str]:
        """Execute the validated query using fake Looker SDK."""
        try:
            # Use model_dump() for Pydantic v2 compatibility
            print("QUERY",query.model_dump())
            result = fake_looker.run_inline_query("json", query.model_dump())
            return result
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return None
    
    def format_response(self, result: str, user_query: str) -> str:
        """Format the query result into a readable response."""
        if not self.gemini_available:
            return result
        
        try:
            data = json.loads(result)
            row_count = len(data)
            
            # Create summary prompt
            prompt = f"""
Based on this query result, provide a brief narrative summary for the user.

Original Question: {user_query}
Number of rows returned: {row_count}
Sample data: {json.dumps(data[:3], indent=2) if data else "No data"}

Provide a concise, business-friendly summary of the results in 1-2 sentences.
"""
            
            response = self.model.generate_content(prompt)
            narrative = response.text.strip()
            
            return f"{narrative}\n\nReturned {row_count} rows."
            
        except Exception as e:
            print(f"⚠️  Could not generate narrative: {e}")
            data = json.loads(result) if result else []
            return f"Query executed successfully. Returned {len(data)} rows."
    
    def chat(self, user_query: str) -> Dict[str, Any]:
        """Main chat function that orchestrates the RAG pipeline."""
        print(f"\n🔍 Processing: '{user_query}'")
        
        # Step 1: Retrieve relevant field documentation
        print("📚 Retrieving relevant fields...")
        field_docs = self.retrieve_relevant_fields(user_query)
        
        # Step 2: Generate query with Gemini
        print("🤖 Generating query...")
        query_dict = self.generate_query(user_query, field_docs)
        
        if not query_dict:
            return {"error": "Failed to generate query"}
        
        print(f"📝 Generated query: {json.dumps(query_dict, indent=2)}")
        
        # Step 3: Validate with Pydantic
        print("✅ Validating query...")
        validated_query = self.validate_query(query_dict)
        
        if not validated_query:
            # Try regenerating once
            print("🔄 Validation failed, attempting to regenerate...")
            query_dict = self.generate_query(user_query, field_docs)
            if query_dict:
                validated_query = self.validate_query(query_dict)
        
        if not validated_query:
            return {"error": "Query validation failed after retry"}
        
        # Step 4: Execute query
        print("🚀 Executing query...")
        result = self.execute_query(validated_query)
        
        if not result:
            return {"error": "Query execution failed"}
        
        # Step 5: Format response
        print("📊 Formatting response...")
        narrative = self.format_response(result, user_query)
        print("NARRATIVE",narrative)
        
        return {
            "success": True,
            "query": validated_query.model_dump(),
            "result": result,
            "narrative": narrative,
            "field_docs": field_docs
        }

def main():
    """Main interactive chat loop."""
    print("🚀 Gemini RAG Chat for LookML Queries")
    print("=" * 50)
    
    # Initialize chat interface
    chat_interface = GeminiRAGChat()
    
    print("\n💡 Example queries:")
    print("  - 'total sales last 7 days'")
    print("  - 'customers by state'")
    print("  - 'top selling product categories'")
    print("  - 'sales by region'")
    print("\n💬 Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("🗣️  Ask: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process query
            response = chat_interface.chat(user_input)
            
            if response.get("error"):
                print(f"❌ Error: {response['error']}")
            else:
                print(f"\n📊 {response['narrative']}")
                
                # Optionally show raw data
                show_data = input("\n🔍 Show raw data? (y/n): ").strip().lower()
                if show_data == 'y':
                    result_data = json.loads(response['result'])
                    print(f"\n📄 Data ({len(result_data)} rows):")
                    print(json.dumps(result_data[:10], indent=2))  # Show first 10 rows
                    if len(result_data) > 10:
                        print(f"... and {len(result_data) - 10} more rows")
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 