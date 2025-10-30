import numpy as np
import math
from collections import Counter

class WordEmbeddings:
    """Proper word embeddings trained from scratch"""
    
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.vocab = {}
    
    def build_embeddings(self, vocabulary):
        """Initialize embeddings"""
        self.vocab = {word: idx for idx, word in enumerate(vocabulary)}
        
        # Initialize with random values scaled by embedding dimension
        scale = math.sqrt(3.0 / self.embedding_dim)
        self.embeddings = np.random.uniform(
            -scale, scale, 
            (len(vocabulary), self.embedding_dim)
        ).astype(np.float32)
    
    def get_embeddings(self, tokens):
        """Get embeddings for tokens"""
        if self.embeddings is None:
            raise ValueError("Embeddings not initialized")
        
        embedded = np.zeros((len(tokens), self.embedding_dim))
        for i, token in enumerate(tokens):
            if token < len(self.embeddings):
                embedded[i] = self.embeddings[token]
        
        return embedded
    
    def train_embeddings(self, sentences, window_size=5, epochs=10, learning_rate=0.025):
        """Train word2vec-style embeddings"""
        # This is a simplified skip-gram implementation
        for epoch in range(epochs):
            total_loss = 0
            for sentence in sentences:
                tokens = [self.vocab.get(word, 0) for word in sentence.split() if word in self.vocab]
                
                for i, target in enumerate(tokens):
                    # Get context words
                    start = max(0, i - window_size)
                    end = min(len(tokens), i + window_size + 1)
                    context = tokens[start:i] + tokens[i+1:end]
                    
                    for context_word in context:
                        # Simplified training step
                        target_vec = self.embeddings[target]
                        context_vec = self.embeddings[context_word]
                        
                        # Simple gradient update (real implementation would use proper backprop)
                        dot_product = np.dot(target_vec, context_vec)
                        gradient = (1 - self.sigmoid(dot_product)) * learning_rate
                        
                        self.embeddings[target] += gradient * context_vec
                        self.embeddings[context_word] += gradient * target_vec
                        
                        total_loss += -math.log(self.sigmoid(dot_product))
            
            print(f"Embedding Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + math.exp(-x))