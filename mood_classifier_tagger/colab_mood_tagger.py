#!/usr/bin/env python3
"""
Google Colab Notebook for Mood Tagging
=====================================

This script runs mood_tagger.py on Google Colab with GPU support.
Processes all 10,000 books in ~15-20 minutes with free T4 GPU.

Usage:
1. Upload this file to Google Colab
2. Run each cell in order
3. Results saved to Google Drive
"""

# ============================================================================
# CELL 1: Mount Google Drive
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive mounted successfully!")

# ============================================================================
# CELL 2: Check GPU and Set Working Directory
# ============================================================================

import torch
import os

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set working directory
os.chdir('/content/drive/MyDrive/Book recommender')
print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir()}")

# ============================================================================
# CELL 3: Install Dependencies
# ============================================================================

!pip install -q transformers sentence-transformers pandas numpy tqdm scikit-learn

print("✅ All dependencies installed!")

# ============================================================================
# CELL 4: Import and Run Mood Tagger
# ============================================================================

import sys
sys.path.insert(0, '/content/drive/MyDrive/Book recommender')

from mood_tagger import tag_books, create_mood_mappings
import time

# Start timing
start_time = time.time()

# Tag all books (or use --limit for testing)
print("\n🚀 Starting mood tagging with GPU acceleration...\n")

df = tag_books(
    csv_path="../dataset/books_enriched.csv",
    output_path="../dataset/books_with_moods.csv",
    top_k=3,
    limit=None  # Set to 100 for testing, None for all books
)

# Save mood mappings
print("\n📊 Creating mood mappings...\n")
mappings = create_mood_mappings(df)

# Calculate elapsed time
elapsed_time = time.time() - start_time
books_per_sec = len(df) / elapsed_time

print(f"\n{'='*80}")
print(f"✅ MOOD TAGGING COMPLETED!")
print(f"{'='*80}")
print(f"Total books: {len(df)}")
print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"Speed: {books_per_sec:.1f} books/second")
print(f"Output file: dataset/books_with_moods.csv")
print(f"Mood mappings: models/mood_to_idx.pkl, models/idx_to_mood.pkl")
print(f"{'='*80}\n")

# ============================================================================
# CELL 5: Verify Results and Download
# ============================================================================

import pandas as pd

# Load and display results
result_df = pd.read_csv("../dataset/books_with_moods.csv")

print(f"✅ Results loaded: {result_df.shape[0]} books tagged\n")
print(f"Columns: {list(result_df.columns)}\n")
print("Sample results:")
print(result_df[['title', 'mood_primary', 'mood_primary_score', 'moods_top3']].head(10))

print(f"\n✅ Files are automatically saved to Google Drive!")
print(f"Location: Book recommender/dataset/books_with_moods.csv")

# ============================================================================

