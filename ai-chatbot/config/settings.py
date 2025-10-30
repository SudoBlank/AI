# Configuration settings for the chatbot

# Model settings
MODEL_CONFIG = {
    'transformer': {
        'model_name': 'gpt2',
        'max_length': 100,
        'temperature': 0.7,
        'num_beams': 5
    },
    'embeddings': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'max_length': 512
    }
}

# Memory settings
MEMORY_CONFIG = {
    'short_term': {
        'max_turns': 10
    },
    'long_term': {
        'db_path': 'data/knowledge/memory.db'
    }
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 10,
    'warmup_steps': 1000,
    'save_steps': 5000,
    'eval_steps': 1000
}

# Server settings
SERVER_CONFIG = {
    'host': 'localhost',
    'port': 3000,
    'debug': True
}

# Paths
PATHS = {
    'model_dir': 'models/checkpoints',
    'data_dir': 'data',
    'logs_dir': 'logs'
}