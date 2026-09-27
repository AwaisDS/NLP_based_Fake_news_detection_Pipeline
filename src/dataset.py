import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    """
    Custom PyTorch Dataset for Fake/True news articles.
    """
    def __init__(self, df, vocab, max_len=300):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'clean_text' and 'label' columns.
            vocab (dict): Word-to-index mapping dictionary.
            max_len (int): Maximum length of token sequences.
        """
        self.labels = df['label'].values
        self.texts = df['clean_text'].values
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.get("<pad>", 0)
        self.unk_idx = vocab.get("<unk>", 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize by space
        tokens = text.split()
        
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        # Truncate if sequence is longer than max_len
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        # Pad with pad_idx if sequence is shorter than max_len
        else:
            padding = [self.pad_idx] * (self.max_len - len(indices))
            indices = indices + padding
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
