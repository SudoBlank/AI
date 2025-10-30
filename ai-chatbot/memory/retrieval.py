import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

class MemoryRetrieval:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def encode_text(self, text):
        """Encode text into vector representation"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings.cpu().numpy()
        
    def compute_similarity(self, query_vec, memory_vecs):
        """Compute cosine similarity between query and memories"""
        # Normalize vectors
        query_norm = np.linalg.norm(query_vec)
        memory_norms = np.linalg.norm(memory_vecs, axis=1)
        
        # Compute cosine similarity
        similarities = np.dot(memory_vecs, query_vec.T) / (memory_norms * query_norm)
        return similarities
        
    def retrieve_relevant(self, query, memories, k=5):
        """Retrieve k most relevant memories for query"""
        # Encode query
        query_vec = self.encode_text(query)
        
        # Encode all memories
        memory_texts = [m['text'] for m in memories]
        memory_vecs = np.vstack([
            self.encode_text(text) for text in memory_texts
        ])
        
        # Get similarities
        similarities = self.compute_similarity(query_vec, memory_vecs)
        
        # Get top k
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        return [
            {
                'memory': memories[idx],
                'similarity': similarities[idx]
            }
            for idx in top_k_idx
        ]