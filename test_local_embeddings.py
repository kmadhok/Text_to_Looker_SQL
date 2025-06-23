#!/usr/bin/env python3
"""
Test script to verify local sentence transformers implementation.
"""

import os
import sys

def test_sentence_transformers():
    """Test sentence transformers installation and basic functionality."""
    print("🧪 Testing Sentence Transformers Implementation")
    print("=" * 50)
    
    try:
        # Test 1: Import sentence-transformers
        print("1️⃣ Testing sentence-transformers import...")
        import sentence_transformers
        print(f"✅ sentence-transformers version: {sentence_transformers.__version__}")
        
        # Test 2: Test HuggingFace embedding with LlamaIndex
        print("\n2️⃣ Testing LlamaIndex HuggingFace integration...")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Test with conservative settings
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            trust_remote_code=False,
            device="cpu"
        )
        print("✅ HuggingFace embedding model initialized")
        
        # Test 3: Generate test embedding
        print("\n3️⃣ Testing embedding generation...")
        test_text = "This is a test document for checking embeddings"
        embedding = embed_model.get_text_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Embedding generated successfully (dimension: {len(embedding)})")
            print(f"📊 Sample values: {embedding[:5]}")
        else:
            print("❌ Embedding generation failed")
            return False
        
        # Test 4: Test with multiple texts
        print("\n4️⃣ Testing batch embedding...")
        test_texts = [
            "customer sales data",
            "product inventory information", 
            "store location details"
        ]
        
        embeddings = []
        for text in test_texts:
            emb = embed_model.get_text_embedding(text)
            embeddings.append(emb)
        
        if all(emb and len(emb) > 0 for emb in embeddings):
            print("✅ Batch embedding successful")
            
            # Calculate similarity (simple dot product)
            import numpy as np
            emb1, emb2 = np.array(embeddings[0]), np.array(embeddings[1])
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"📏 Similarity between texts: {similarity:.3f}")
        else:
            print("❌ Batch embedding failed")
            return False
        
        print("\n🎉 All tests passed! Local embeddings are working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your environment and dependencies")
        return False

def test_minimal_fallback():
    """Test the minimal keyword-based fallback."""
    print("\n🔄 Testing Keyword-Based Fallback")
    print("=" * 35)
    
    try:
        # Create sample documents
        sample_docs = {
            "customers.customer_id": "Unique identifier for each customer in the system",
            "orders.order_date": "Date when the order was placed by the customer",
            "products.category": "Product category like Electronics, Clothing, etc.",
            "order_items.total_sales": "Total sales amount for order items"
        }
        
        # Test keyword search
        query = "sales revenue"
        keywords = query.lower().split()
        
        relevant_fields = []
        for field_name, description in sample_docs.items():
            content = description.lower()
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                relevant_fields.append((score, field_name, description))
        
        relevant_fields.sort(key=lambda x: x[0], reverse=True)
        
        print(f"🔍 Query: '{query}'")
        print("📋 Relevant fields:")
        for score, field_name, description in relevant_fields:
            print(f"  {score}⭐ {field_name}: {description}")
        
        if relevant_fields:
            print("✅ Keyword search working correctly")
            return True
        else:
            print("❌ No relevant fields found")
            return False
            
    except Exception as e:
        print(f"❌ Keyword search error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 LOCAL EMBEDDING SYSTEM TEST")
    print("=" * 60)
    
    # Suppress GOOGLE_API_KEY for this test
    original_key = os.environ.get('GOOGLE_API_KEY')
    if 'GOOGLE_API_KEY' in os.environ:
        del os.environ['GOOGLE_API_KEY']
    
    try:
        success = True
        
        # Test sentence transformers
        if not test_sentence_transformers():
            success = False
        
        # Test fallback
        if not test_minimal_fallback():
            success = False
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Local embeddings are ready for use")
            print("💡 Your system can work without OpenAI API keys")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("⚠️  Check installation and try troubleshooting steps")
            
    finally:
        # Restore original API key
        if original_key:
            os.environ['GOOGLE_API_KEY'] = original_key
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 