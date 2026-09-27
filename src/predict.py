import os
import json
import torch
from src import config
from src.data_preprocessing import clean_text
from src.model import BiLSTMClassifier

class NewsPredictor:
    """
    Predictor class to classify raw news articles as FAKE or TRUE.
    """
    def __init__(self):
        model_save_path = os.path.join(config.MODEL_DIR, "best_model.pt")
        
        # 1. Validation checks
        if not os.path.exists(config.VOCAB_PATH):
            raise FileNotFoundError(f"Vocabulary file not found at '{config.VOCAB_PATH}'. Please train the model first.")
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Trained model checkpoint not found at '{model_save_path}'. Please train the model first.")
            
        # 2. Load vocab
        with open(config.VOCAB_PATH, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.pad_idx = self.vocab.get("<pad>", 0)
        self.unk_idx = self.vocab.get("<unk>", 1)
        
        # 3. Load model structure and weights
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            bidirectional=config.BIDIRECTIONAL
        ).to(config.DEVICE)
        
        self.model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
        self.model.eval()
        
    def predict(self, text):
        """
        Predicts whether a news text is FAKE or TRUE.
        
        Returns:
            label (str): "TRUE" or "FAKE"
            confidence (float): Probability score of the predicted class
        """
        # Preprocess and clean text
        cleaned = clean_text(text)
        tokens = cleaned.split()
        
        # Numericalize tokens
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        # Pad and truncate
        if len(indices) > config.MAX_LEN:
            indices = indices[:config.MAX_LEN]
        else:
            padding = [self.pad_idx] * (config.MAX_LEN - len(indices))
            indices = indices + padding
            
        # Convert to tensor and batch dimension (1, MAX_LEN)
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(config.DEVICE)
        
        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits).item()
            
        # Classification threshold
        if probability >= 0.5:
            label = "TRUE"
            confidence = probability
        else:
            label = "FAKE"
            confidence = 1.0 - probability
            
        return label, confidence

def interactive_cli():
    print("=" * 60)
    print("       FAKE VS TRUE NEWS DETECTION - PREDICTOR CLI")
    print("=" * 60)
    
    try:
        predictor = NewsPredictor()
        print("Model loaded successfully. Ready for predictions.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("\nInstructions: Paste your news article title & content, then press Enter.")
    print("Type 'exit' or 'quit' to terminate the session.\n")
    
    while True:
        try:
            user_input = input("Enter News Article Text:\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting Predictor CLI. Goodbye!")
                break
                
            label, confidence = predictor.predict(user_input)
            
            print("\n" + "-" * 40)
            print(f"Classification Result: {label}")
            print(f"Confidence Level:      {confidence * 100:.2f}%")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting Predictor CLI. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    interactive_cli()
