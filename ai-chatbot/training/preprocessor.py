import torch
import numpy as np
from transformers import AutoTokenizer
import json
import os

class DataPreprocessor:
    def __init__(self, tokenizer_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_raw_data(self, input_file):
        """Load raw scraped data"""
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def clean_text(self, text):
        """Clean and normalize text"""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def tokenize(self, text, max_length=512):
        """Tokenize text using the pretrained tokenizer"""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
    def prepare_training_data(self, raw_data, output_dir='data/processed'):
        """Process raw data into training format"""
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = []
        for url, content in raw_data.items():
            clean_content = self.clean_text(content)
            tokens = self.tokenize(clean_content)
            
            processed_data.append({
                'url': url,
                'content': clean_content,
                'input_ids': tokens['input_ids'].tolist(),
                'attention_mask': tokens['attention_mask'].tolist()
            })
            
        # Save processed data
        output_file = os.path.join(output_dir, 'processed_data.json')
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        return processed_data
        
    def create_dataloaders(self, processed_data, batch_size=32):
        """Create PyTorch dataloaders from processed data"""
        # Convert to tensors
        input_ids = torch.tensor([d['input_ids'][0] for d in processed_data])
        attention_masks = torch.tensor([d['attention_mask'][0] for d in processed_data])
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
        
        # Create dataloaders
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        return train_dataloader, val_dataloader