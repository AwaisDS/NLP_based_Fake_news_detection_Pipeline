import os
import json
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import config
from src.data_preprocessing import preprocess_pipeline
from src.dataset import NewsDataset
from src.model import BiLSTMClassifier

def calculate_accuracy(preds, labels):
    """
    Computes accuracy for a batch of binary predictions.
    """
    # Round predictions to 0 or 1 using sigmoid on logits
    probs = torch.sigmoid(preds)
    predictions = (probs >= 0.5).float()
    correct = (predictions == labels).float()
    return correct.sum() / len(correct)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass & Optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc = calculate_accuracy(outputs, labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def run_training():
    print(f"Using device: {config.DEVICE}")
    
    # 1. Trigger preprocessing if datasets don't exist
    train_path = os.path.join(config.PROCESSED_DIR, "train.csv")
    val_path = os.path.join(config.PROCESSED_DIR, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(config.VOCAB_PATH):
        print("Processed files or vocabulary not found. Triggering preprocessing pipeline first...")
        preprocess_pipeline()
        
    # Load vocabulary
    with open(config.VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    # Load dataframes
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Drop any null rows
    train_df = train_df.dropna().reset_index(drop=True)
    val_df = val_df.dropna().reset_index(drop=True)
    
    # 2. Create datasets & dataloaders
    train_dataset = NewsDataset(train_df, vocab, max_len=config.MAX_LEN)
    val_dataset = NewsDataset(val_df, vocab, max_len=config.MAX_LEN)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    # 3. Initialize Model, Criterion, and Optimizer
    vocab_size = len(vocab)
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    ).to(config.DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 4. Training Loop
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    best_val_loss = float('inf')
    model_save_path = os.path.join(config.MODEL_DIR, "best_model.pt")
    
    print("\n--- Training Model ---")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        # Save metrics history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{config.EPOCHS:02d} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
        
        # Checkpoint: save model with lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model checkpoint to {model_save_path}")
            
    print(f"\nTraining completed! Best Validation Loss: {best_val_loss:.4f}")
    
    # 5. Plot Loss & Accuracy Curves
    plot_curves(history)

def plot_curves(history):
    """
    Plots training and validation curves.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="orange")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc", color="royalblue")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc", color="orange")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(config.PLOTS_DIR, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved learning curve plots to: {plot_path}")

if __name__ == "__main__":
    run_training()
