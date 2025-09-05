#!/usr/bin/env python3
"""
Test script for the MxBai Reranker Service
"""

import requests
import json

# Test data
query = "Who wrote 'To Kill a Mockingbird'?"
documents = [
    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
    "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
    "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
    "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
    "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
]

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:8001/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_rerank():
    """Test rerank endpoint"""
    try:
        payload = {
            "query": query,
            "documents": documents,
            "top_k": 3,
            "return_documents": True
        }
        
        response = requests.post("http://localhost:8001/rerank", json=payload)
        print(f"\nRerank Test: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Total documents processed: {result['total_documents']}")
            print(f"Query: {result['query']}")
            print("\nTop results:")
            
            for i, item in enumerate(result['results']):
                print(f"{i+1}. Score: {item['score']:.4f}")
                print(f"   Index: {item['index']}")
                if 'document' in item:
                    print(f"   Document: {item['document'][:100]}...")
                print()
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Rerank test failed: {e}")

def test_simple_rerank():
    """Test simple rerank endpoint"""
    try:
        payload = {
            "query": query,
            "documents": documents,
            "top_k": 3
        }
        
        response = requests.post("http://localhost:8001/rerank/simple", json=payload)
        print(f"\nSimple Rerank Test: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Query: {result['query']}")
            print("Scores:", result['scores'])
            print("Indices:", result['indices'])
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Simple rerank test failed: {e}")

if __name__ == "__main__":
    print("Testing MxBai Reranker Service...")
    print("=" * 50)
    
    # Test health first
    if test_health():
        print("\n" + "=" * 50)
        test_rerank()
        print("\n" + "=" * 50)
        test_simple_rerank()
    else:
        print("Service is not healthy. Make sure the server is running on port 8001.")
