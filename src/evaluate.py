import os
import json
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src import config
from src.dataset import NewsDataset
from src.model import BiLSTMClassifier

def evaluate_model():
    print(f"Using device: {config.DEVICE}")
    
    test_path = os.path.join(config.PROCESSED_DIR, "test.csv")
    model_save_path = os.path.join(config.MODEL_DIR, "best_model.pt")
    
    # 1. Validation checks
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at '{test_path}'. Please run preprocessing first.")
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Trained model not found at '{model_save_path}'. Please run training first.")
    if not os.path.exists(config.VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary file not found at '{config.VOCAB_PATH}'.")

    # 2. Load vocab and test dataset
    with open(config.VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    test_df = pd.read_csv(test_path)
    test_df = test_df.dropna().reset_index(drop=True)
    
    test_dataset = NewsDataset(test_df, vocab, max_len=config.MAX_LEN)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. Load model checkpoint
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
    model.eval()
    
    # 4. Predict on test set
    all_predictions = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions = (probs >= 0.5).int().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.int().numpy())
            
    # 5. Calculate metrics
    print("\n--- Classification Report ---")
    report = classification_report(all_labels, all_predictions, target_names=["FAKE", "TRUE"], digits=4)
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Save metrics report as text
    report_path = os.path.join(config.PLOTS_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved evaluation text report to: {report_path}")
    
    # 6. Plot Confusion Matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Predicted FAKE", "Predicted TRUE"],
        yticklabels=["Actual FAKE", "Actual TRUE"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    
    cm_plot_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

if __name__ == "__main__":
    evaluate_model()
