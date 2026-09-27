import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier for Binary Text Classification.
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=True):
        super(BiLSTMClassifier, self).__init__()
        
        # Word Embeddings (index 0 is for padding)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM Layer
        # If bidirectional is True, output features will be hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Bidirectional scaling
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Combine global average pooling and global max pooling
        # Therefore, input to FC is lstm_out_dim * 2
        fc_input_dim = lstm_out_dim * 2
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single output logit for binary classification
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        
        # 1. Embedding layer
        embedded = self.embedding(x)  # shape: [batch_size, sequence_length, embedding_dim]
        
        # 2. LSTM layer
        lstm_out, _ = self.lstm(embedded)  # shape: [batch_size, sequence_length, lstm_out_dim]
        
        # 3. Global Pooling (Average and Max pooling over the sequence dimension)
        # Average pooling: shape [batch_size, lstm_out_dim]
        avg_pool = torch.mean(lstm_out, dim=1)
        
        # Max pooling: shape [batch_size, lstm_out_dim]
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Concatenate average and max pool outputs
        pooled_out = torch.cat((avg_pool, max_pool), dim=1)  # shape: [batch_size, lstm_out_dim * 2]
        
        # 4. Dense classification layer
        logits = self.fc(pooled_out)  # shape: [batch_size, 1]
        
        return logits.squeeze(1)  # shape: [batch_size]
