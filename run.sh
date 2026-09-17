#!/bin/bash
echo "=================================================="
echo "   Starting Water Quality Anomaly Detection CLI"
echo "=================================================="
echo ""

if [ ! -f "venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Please run: python3 -m venv venv"
    echo "and install dependencies using: pip install -r requirements.txt"
    exit 1
fi

venv/bin/python main.py
