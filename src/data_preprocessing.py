import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

def clean_text(text):
    """
    Cleans raw text by:
    - Converting to lowercase
    - Removing URLs and HTML tags
    - Removing non-alphabetic characters
    - Collapsing multiple spaces
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Keep only alphabetic characters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_merge_data():
    """
    Loads Fake.csv and True.csv, adds labels, and merges them.
    """
    print("Loading datasets...")
    if not os.path.exists(config.FAKE_DATA_PATH) or not os.path.exists(config.TRUE_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset files not found. Ensure '{config.FAKE_DATA_PATH}' and '{config.TRUE_DATA_PATH}' exist."
        )
        
    fake_df = pd.read_csv(config.FAKE_DATA_PATH)
    true_df = pd.read_csv(config.TRUE_DATA_PATH)
    
    # Add label column: 0 for Fake, 1 for True
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine title and text for rich context
    print("Combining title and text...")
    fake_df['full_text'] = fake_df['title'] + " " + fake_df['text']
    true_df['full_text'] = true_df['title'] + " " + true_df['text']
    
    # Keep only relevant columns
    fake_df = fake_df[['full_text', 'label']]
    true_df = true_df[['full_text', 'label']]
    
    # Concatenate
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    return combined_df

def build_vocab(texts, max_vocab_size=50000, min_freq=2):
    """
    Builds a vocabulary mapping from a list of cleaned texts.
    Special tokens: <pad> -> 0, <unk> -> 1
    """
    print("Building vocabulary...")
    word_counts = {}
    
    for text in texts:
        tokens = text.split()
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
            
    # Filter words by frequency and sort by frequency
    vocab_list = [word for word, count in word_counts.items() if count >= min_freq]
    vocab_list = sorted(vocab_list, key=lambda w: word_counts[w], reverse=True)
    
    # Limit vocab size
    vocab_list = vocab_list[:max_vocab_size - 2]
    
    # Create vocabulary mapping
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, word in enumerate(vocab_list):
        vocab[word] = idx + 2
        
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def preprocess_pipeline():
    """
    Full pipeline to clean, split, build vocab, and save data.
    """
    print("Starting preprocessing pipeline...")
    
    # 1. Load and merge
    df = load_and_merge_data()
    
    # 2. Clean text
    print("Cleaning text (this may take a minute)...")
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    # Drop rows with empty cleaned text
    df = df[df['clean_text'] != ""].reset_index(drop=True)
    
    # 3. Train / Val / Test Split (80% Train, 10% Val, 10% Test)
    print("Splitting datasets...")
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=config.RANDOM_STATE, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=config.RANDOM_STATE, stratify=temp_df['label']
    )
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    
    # 4. Build vocabulary using training texts ONLY (to avoid data leakage)
    vocab = build_vocab(train_df['clean_text'].tolist())
    
    # 5. Save outputs
    print(f"Saving splits to {config.PROCESSED_DIR}...")
    train_df.to_csv(os.path.join(config.PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.PROCESSED_DIR, "test.csv"), index=False)
    
    # Save vocabulary
    with open(config.VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
        
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_pipeline()
