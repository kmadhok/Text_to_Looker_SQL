#!/usr/bin/env python3
"""
Build FAISS index from LookML field documentation for semantic search.
Uses Google's Gemini embedding model to create vector embeddings of field descriptions.
"""

import os
import pickle
from pathlib import Path
import google.generativeai as genai
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.keyword_table import SimpleKeywordTableIndex
import json

# Load environment variables from .env file if it exists
load_dotenv()

def setup_gemini():
    """Setup Google Gemini API (requires GOOGLE_API_KEY environment variable)."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_API_KEY environment variable not set.")
        print("📝 You can set it with: export GOOGLE_API_KEY='your-api-key'")
        print("📝 Or create a .env file with: GOOGLE_API_KEY=your-api-key")
        print("🔑 Get an API key from: https://aistudio.google.com/app/apikey")
        print("📋 For now, creating a dummy index that can be replaced later...")
        return None
    
    genai.configure(api_key=api_key)
    print("✅ Google API key loaded successfully")
    return api_key

def load_field_docs():
    """Load all field documentation from lookml_docs directory."""
    docs_dir = Path("lookml_docs")
    documents = []
    
    if not docs_dir.exists():
        print(f"❌ Directory {docs_dir} not found!")
        return documents
    
    for doc_file in docs_dir.glob("*.txt"):
        try:
            content = doc_file.read_text(encoding='utf-8')
            # Extract field name from filename (e.g., customers.customer_id.txt -> customers.customer_id)
            field_name = doc_file.stem
            
            # Create document with metadata
            doc = Document(
                text=content,
                metadata={
                    "field_name": field_name,
                    "file_path": str(doc_file),
                    "doc_type": "field_documentation"
                }
            )
            documents.append(doc)
            print(f"📄 Loaded: {field_name}")
            
        except Exception as e:
            print(f"❌ Error loading {doc_file}: {e}")
    
    return documents

def create_dummy_index(documents):
    """Create a dummy index when Google API key is not available."""
    print("🔧 Creating local index without external dependencies...")
    
    # Skip sentence transformers for now due to segfault issues on this system
    # Go directly to keyword-based indexing which is more reliable
    
    # Strategy 1: Keyword-based index (most reliable)
    try:
        print("🔧 Creating keyword-based index...")
        keyword_index = SimpleKeywordTableIndex.from_documents(documents)
        print("✅ Successfully created keyword-based index")
        return keyword_index
        
    except Exception as e:
        print(f"⚠️  Keyword index failed: {e}")
    
    # Strategy 2: Minimal document store (ultimate fallback)
    try:
        print("🆘 Creating minimal document store...")
        
        # Create directory
        os.makedirs("faiss_index", exist_ok=True)
        
        # Save documents as simple JSON
        doc_store = {}
        for i, doc in enumerate(documents):
            doc_store[i] = {
                "content": doc.text,
                "metadata": doc.metadata
            }
        
        with open("faiss_index/keyword_docs.json", "w") as f:
            json.dump(doc_store, f, indent=2)
        
        print("📄 Created minimal keyword-searchable document store")
        return "keyword_store"  # Special indicator
        
    except Exception as e:
        print(f"❌ All indexing strategies failed: {e}")
        print("💡 System working with Google API key for full functionality")
        return None

def create_dummy_index_with_embeddings(documents):
    """Alternative function for sentence transformers when system supports it."""
    print("🔧 Creating local index with sentence transformers...")
    
    # This function can be called separately when the system is stable
    try:
        print("🤖 Attempting sentence transformers embedding...")
        
        # Test if embedding works first in isolation
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            trust_remote_code=False,
            device="cpu",
            max_length=256,
            normalize=True
        )
        
        # Quick test
        test_embedding = embed_model.get_text_embedding("quick test")
        if not test_embedding:
            raise ValueError("Test embedding failed")
        
        print("✅ Embedding model test passed, creating index...")
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=embed_model,
            show_progress=False
        )
        
        print("✅ Successfully created vector index with sentence transformers")
        return index
        
    except Exception as e:
        print(f"⚠️  Sentence transformers failed: {e}")
        return None

def build_index():
    """Build FAISS index from field documentation."""
    print("🔄 Building FAISS index from field documentation...")
    
    # Setup Gemini API
    api_key = setup_gemini()
    
    # Load field documentation
    documents = load_field_docs()
    
    if not documents:
        print("❌ No documents found to index!")
        return None
    
    print(f"📚 Loaded {len(documents)} field documentation files")
    
    try:
        if api_key:
            # Use Google Gemini embeddings
            print("🔗 Using Google Gemini embeddings...")
            embed_model = GoogleGenAIEmbedding(
                model_name="models/embedding-001",
                api_key=api_key
            )
            
            # Create FAISS vector store
            dimension = 768  # Gemini embedding dimension
            faiss_index = faiss.IndexFlatL2(dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # Build index with custom embedding model
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store,
                embed_model=embed_model
            )
            
        else:
            # Create dummy index for development
            index = create_dummy_index(documents)
            if not index:
                return None
            
            # Handle special case for keyword_store
            if index == "keyword_store":
                print("💾 Document store created successfully")
                
                # Create minimal index structure for compatibility
                doc_metadata = []
                for doc in documents:
                    doc_metadata.append({
                        "field_name": doc.metadata["field_name"],
                        "content": doc.text
                    })
                
                with open("faiss_index/doc_metadata.pkl", "wb") as f:
                    pickle.dump(doc_metadata, f)
                
                print("✅ Local document search system built successfully!")
                print(f"📁 Files saved to ./faiss_index/")
                print(f"📊 Indexed {len(documents)} field documents")
                print("⚠️  Note: Using keyword-based search. Set GOOGLE_API_KEY for semantic search.")
                
                return "keyword_store"  # Return the indicator
        
        # Save the index (only for real indices, not keyword_store)
        if index != "keyword_store":
            print("💾 Saving index to disk...")
            index.storage_context.persist(persist_dir="./faiss_index")
        
        # Also save document metadata for reference
        doc_metadata = []
        for doc in documents:
            doc_metadata.append({
                "field_name": doc.metadata["field_name"],
                "content": doc.text
            })
        
        with open("faiss_index/doc_metadata.pkl", "wb") as f:
            pickle.dump(doc_metadata, f)
        
        print("✅ FAISS index built successfully!")
        print(f"📁 Index saved to ./faiss_index/")
        print(f"📊 Indexed {len(documents)} field documents")
        
        if not api_key:
            print("⚠️  Note: This is a dummy index. Set GOOGLE_API_KEY and rebuild for full functionality.")
        
        return index
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        if "API_KEY" in str(e):
            print("💡 Try setting GOOGLE_API_KEY environment variable")
        return None

def test_index():
    """Test the built index with a sample query."""
    print("\n🧪 Testing index with sample query...")
    
    try:
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        
        # Determine which embedding model to use for testing
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if api_key:
            # Use Google Gemini embeddings (same as when building)
            embed_model = GoogleGenAIEmbedding(
                model_name="models/embedding-001",
                api_key=api_key
            )
            print("🔗 Using Google Gemini embeddings for testing...")
        else:
            # Use local embeddings for testing
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                trust_remote_code=True,
                device="cpu"
            )
            print("🤖 Using local sentence transformers for testing...")
        
        # Load the index with the appropriate embedding model
        storage_context = StorageContext.from_defaults(persist_dir="./faiss_index")
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="no_text"  # Just return relevant documents
        )
        
        # Test query
        test_query = "fields for sales analysis and revenue"
        print(f"🔍 Query: '{test_query}'")
        
        response = query_engine.query(test_query)
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("📋 Top relevant fields:")
            for i, node in enumerate(response.source_nodes[:3], 1):
                field_name = node.metadata.get('field_name', 'Unknown')
                score = getattr(node, 'score', 0)
                print(f"  {i}. {field_name} (score: {score:.3f})")
        else:
            print("📄 Index test completed (no detailed results available)")
        
        print("✅ Index test successful!")
        
    except Exception as e:
        print(f"❌ Index test failed: {e}")
        # Try a simpler test without embeddings
        try:
            print("🔄 Trying basic index load without embeddings...")
            storage_context = StorageContext.from_defaults(persist_dir="./faiss_index")
            index = load_index_from_storage(storage_context)
            print("✅ Basic index load successful!")
        except Exception as e2:
            print(f"❌ Basic index load also failed: {e2}")

def main():
    """Main function to build the index."""
    print("🚀 FAISS Index Builder for LookML Field Documentation")
    print("=" * 60)
    
    # Create directories
    os.makedirs("faiss_index", exist_ok=True)
    
    # Build index
    index = build_index()
    
    if index:
        # Only test real indices, not keyword_store
        if index != "keyword_store":
            # Test the index
            test_index()
        
        print("\n🎉 Index building complete!")
        print("💡 You can now use chat_gemini.py to query the indexed fields")
    else:
        print("\n❌ Index building failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 