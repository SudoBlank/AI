#!/usr/bin/env python3
"""
Main script to train the AI from scratch with real data
"""

import argparse
from training.scraper import KnowledgeScraper
from training.trainer import RealAITrainer
import time
import os

def main():
    parser = argparse.ArgumentParser(description='Train AI from scratch')
    parser.add_argument('--scrape-hours', type=int, default=24, help='Hours to spend scraping')
    parser.add_argument('--train-epochs', type=int, default=100, help='Epochs to train')
    parser.add_argument('--skip-scraping', action='store_true', help='Skip scraping phase')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if not args.skip_scraping:
        print("=== PHASE 1: SCRAPING REAL KNOWLEDGE ===")
        scraper = KnowledgeScraper()
        scraper.run_scraping_session(hours=args.scrape_hours)
        print(f"Scraping completed! Collected {len(scraper.knowledge_base)} knowledge entries.")
    
    print("=== PHASE 2: TRAINING AI MODEL ===")
    trainer = RealAITrainer()
    trainer.load_knowledge('data/knowledge_base.json')
    trainer.train(epochs=args.train_epochs)
    
    print("=== TRAINING COMPLETED ===")
    print("Your AI is now trained on real data and should be much smarter!")

if __name__ == "__main__":
    main()