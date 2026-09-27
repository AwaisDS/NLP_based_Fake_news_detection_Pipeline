import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data paths
FAKE_DATA_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_DATA_PATH = os.path.join(DATA_DIR, "True.csv")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.json")

# Hyperparameters
MAX_LEN = 300            # Max sequence length for tokenization
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# Model Architecture Parameters
EMBEDDING_DIM = 100       # Size of vocabulary embedding
HIDDEN_DIM = 128          # Hidden units in LSTM
NUM_LAYERS = 2            # Number of LSTM layers
DROPOUT = 0.3             # Dropout rate for regularization
BIDIRECTIONAL = True      # Use bidirectional LSTM

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
