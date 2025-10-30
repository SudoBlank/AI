import numpy as np
import json
import pickle
from models.neural_core import NeuralCore, Tensor
from models.embeddings import WordEmbeddings
import time
from collections import Counter
import os

class RealAITrainer:
    """Trains the AI on real scraped data"""
    
    def __init__(self):
        self.vocab_size = 50000
        self.embedding_dim = 512
        self.hidden_dim = 1024
        self.batch_size = 32
        self.learning_rate = 0.001
        
        self.embeddings = WordEmbeddings(self.vocab_size, self.embedding_dim)
        self.model = NeuralCore([
            self.embedding_dim * 2,  # Question + context
            self.hidden_dim,
            self.hidden_dim // 2,
            self.vocab_size  # Output: next word probabilities
        ])
        
        self.vocab = {}
        self.inverse_vocab = {}
        self.training_data = []
    
    def load_knowledge(self, filename='data/knowledge_base.json'):
        """Load scraped knowledge for training"""
        with open(filename, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        # Convert to training pairs
        for item in knowledge:
            question = item['question']
            answer = item['answer']
            self.training_data.append((question, answer))
        
        print(f"Loaded {len(self.training_data)} training pairs")
    
    def build_vocabulary(self):
        """Build vocabulary from training data"""
        word_counts = Counter()
        for question, answer in self.training_data:
            words = question.split() + answer.split()
            word_counts.update(words)
        
        # Most common words
        common_words = word_counts.most_common(self.vocab_size - 4)  # Reserve spots for special tokens
        
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<START>': 2,
            '<END>': 3
        }
        
        for i, (word, count) in enumerate(common_words):
            self.vocab[word] = i + 4
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Initialize embeddings
        self.embeddings.build_embeddings(list(self.vocab.keys()))
    
    def text_to_tokens(self, text, max_length=128):
        """Convert text to token indices"""
        tokens = [self.vocab.get('<START>')]
        words = text.split()[:max_length-2]  # Reserve space for START/END
        
        for word in words:
            tokens.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        tokens.append(self.vocab.get('<END>'))
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.vocab['<PAD>'])
        
        return tokens[:max_length]
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        np.random.shuffle(self.training_data)
        
        for i in range(0, len(self.training_data), self.batch_size):
            batch = self.training_data[i:i + self.batch_size]
            if len(batch) < self.batch_size:
                continue
            
            batch_loss = self.train_batch(batch)
            total_loss += batch_loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {batch_loss:.4f}")
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train_batch(self, batch):
        """Train on a single batch"""
        self.model.zero_grad()
        batch_loss = 0
        
        for question, answer in batch:
            # Convert to embeddings
            q_tokens = self.text_to_tokens(question)
            a_tokens = self.text_to_tokens(answer)
            
            # Get embeddings
            q_emb = self.embeddings.get_embeddings(q_tokens)
            a_emb = self.embeddings.get_embeddings(a_tokens)
            
            # Combine question and answer for context
            context = np.concatenate([q_emb.flatten(), a_emb.flatten()])
            context_tensor = Tensor(context.reshape(1, -1))
            
            # Forward pass (simplified - real implementation would be more complex)
            output = self.model(context_tensor)
            
            # Calculate loss (cross-entropy simplified)
            target = Tensor(self.text_to_tokens(answer))
            loss = self.cross_entropy_loss(output, target)
            
            # Backward pass
            loss.backward()
            batch_loss += loss.data
        
        # Update weights
        self.update_weights()
        return batch_loss / len(batch)
    
    def cross_entropy_loss(self, output, target):
        """Simplified cross-entropy loss"""
        # This is a simplified version - real implementation would be more complex
        probs = self.softmax(output.data)
        target_idx = int(target.data[0, 0]) if target.data.size > 0 else 0
        loss = -np.log(probs[0, target_idx] + 1e-8)
        return Tensor([loss])
    
    def softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def update_weights(self):
        """Update model weights using gradient descent"""
        for param in self.model.parameters():
            if param.requires_grad:
                param.data -= self.learning_rate * param.grad
    
    def train(self, epochs=100, save_interval=10):
        """Main training loop"""
        print("Starting training...")
        self.build_vocabulary()
        
        for epoch in range(epochs):
            start_time = time.time()
            avg_loss = self.train_epoch(epoch)
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            
            if (epoch + 1) % save_interval == 0:
                self.save_model(f'models/ai_model_epoch_{epoch+1}.pkl')
        
        print("Training completed!")
    
    def save_model(self, filename):
        """Save trained model"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'embeddings': self.embeddings
            }, f)
        print(f"Model saved to {filename}")