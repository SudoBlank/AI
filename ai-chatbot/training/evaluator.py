import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def evaluate(self, dataloader):
        """Evaluate model performance"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() 
                         if isinstance(v, torch.Tensor)}
                
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(inputs['labels'].cpu().numpy())
                
        return self.compute_metrics(all_preds, all_labels)
        
    def compute_metrics(self, preds, labels):
        """Compute evaluation metrics"""
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def plot_confusion_matrix(self, dataloader, save_path=None):
        """Plot confusion matrix of model predictions"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() 
                         if isinstance(v, torch.Tensor)}
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(inputs['labels'].cpu().numpy())
                
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_report(self, dataloader, save_dir='reports'):
        """Generate comprehensive evaluation report"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get metrics
        metrics = self.evaluate(dataloader)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            dataloader,
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Save metrics report
        report = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__
        }
        
        with open(os.path.join(save_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
            
        return report