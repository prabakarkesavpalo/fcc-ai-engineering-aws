#!/usr/bin/env python3
"""
Test script to verify all components are working correctly
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import psycopg2
        print("✅ psycopg2 imported successfully")
    except ImportError as e:
        print(f"❌ psycopg2 import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ google.generativeai imported successfully")
    except ImportError as e:
        print(f"❌ google.generativeai import failed: {e}")
        return False
        
    try:
        from pgvector.psycopg2 import register_vector
        print("✅ pgvector imported successfully")
    except ImportError as e:
        print(f"❌ pgvector import failed: {e}")
        return False
    
    try:
        import pymupdf
        print("✅ pymupdf imported successfully")
    except ImportError as e:
        print(f"❌ pymupdf import failed: {e}")
        return False
        
    return True

def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing database connection...")
    
    try:
        import psycopg2
        
        # Database configuration
        possible_users = [
            os.getenv('USER'),
            'postgres',
            os.getenv('DB_USER'),
        ]
        possible_users = [user for user in possible_users if user]
        
        DB_CONFIG = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'multimodal_rag'),
            'user': possible_users[0] if possible_users else 'postgres',
            'password': os.getenv('DB_PASSWORD', ''),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        print(f"📝 Database config: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        print("✅ Database connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_google_api():
    """Test Google API configuration"""
    print("\n🔍 Testing Google API configuration...")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_API_KEY environment variable not set")
        print("   This is expected if you haven't set it yet")
        return True
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try to list models to test the API key
        models = list(genai.list_models())
        if models:
            print(f"✅ Google API key is valid - found {len(models)} models")
            return True
        else:
            print("⚠️  Google API key might be invalid - no models returned")
            return False
            
    except Exception as e:
        print(f"❌ Google API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running setup verification tests...\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Database Connection", test_database_connection),
        ("Google API", test_google_api),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("="*60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
