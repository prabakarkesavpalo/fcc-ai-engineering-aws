#!/bin/bash

# Multimodal RAG Local Setup Runner
# This script activates the virtual environment and runs the Python script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run the following commands first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  GOOGLE_API_KEY environment variable is not set"
    echo "   You will be prompted to enter it when running the script"
    echo ""
fi

# Run the Python script
echo "🚀 Starting Multimodal RAG Local system..."
echo "   Using virtual environment: $(which python)"
echo ""

python 02_Multi_modal_RAG_local.py
