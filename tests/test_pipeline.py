import unittest
import torch
import pandas as pd
from src.data_preprocessing import clean_text
from src.dataset import NewsDataset
from src.model import BiLSTMClassifier

class TestNLPNewsPipeline(unittest.TestCase):
    
    def test_text_cleaning(self):
        """
        Verify that raw text is correctly cleaned: URLs removed, HTML stripped,
        and strings converted to lowercase with single space separation.
        """
        raw_text = "<html><body>Hey! Visit https://example.com for more news.  This is a   TEST.</body></html>"
        expected_cleaned = "hey visit for more news this is a test"
        
        cleaned = clean_text(raw_text)
        self.assertEqual(cleaned, expected_cleaned)
        
    def test_dataset_padding_and_truncation(self):
        """
        Verify that NewsDataset correctly encodes, pads, and truncates text sequences to MAX_LEN.
        """
        vocab = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}
        
        # Test case 1: Sequence shorter than max_len (needs padding)
        df_short = pd.DataFrame({"clean_text": ["hello world"], "label": [1]})
        dataset_short = NewsDataset(df_short, vocab, max_len=5)
        indices, label = dataset_short[0]
        
        self.assertEqual(len(indices), 5)
        self.assertEqual(indices[0].item(), 2)  # 'hello'
        self.assertEqual(indices[1].item(), 3)  # 'world'
        self.assertEqual(indices[2].item(), 0)  # padding
        self.assertEqual(label.item(), 1.0)
        
        # Test case 2: Sequence longer than max_len (needs truncation)
        df_long = pd.DataFrame({"clean_text": ["hello world hello world hello"], "label": [0]})
        dataset_long = NewsDataset(df_long, vocab, max_len=3)
        indices, label = dataset_long[0]
        
        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[0].item(), 2)  # 'hello'
        self.assertEqual(indices[1].item(), 3)  # 'world'
        self.assertEqual(indices[2].item(), 2)  # 'hello'
        self.assertEqual(label.item(), 0.0)

    def test_model_shapes(self):
        """
        Verify that the BiLSTMClassifier forward pass returns output predictions of correct shape.
        """
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=16,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=True
        )
        
        # Mock input batch of shape [batch_size, seq_len]
        mock_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        
        # Get predictions
        outputs = model(mock_input)
        
        # The output shape should be [batch_size]
        self.assertEqual(outputs.shape, (batch_size,))

if __name__ == "__main__":
    unittest.main()
