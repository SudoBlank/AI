import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(self.pos_encoder, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def forward(self, src, src_mask=None):
        # src shape: [seq_len, batch_size]
        
        # Embed and scale
        src = self.embedding(src) * np.sqrt(self.d_model)
        
        # Add positional encoding
        output = self.transformer(src, src_mask)
        
        # Decode
        output = self.decoder(output)
        return output
        
    def generate(self, input_ids, max_length=100):
        """Generate sequence using trained transformer"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                outputs = self.forward(input_ids)
                next_token = outputs[-1].argmax(dim=-1)
                
                # Append and continue
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)])
                
                # Stop if we predict the end token
                if next_token.item() == self.end_token:
                    break
                    
        return input_ids