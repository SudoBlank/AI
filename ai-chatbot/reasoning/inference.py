import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class InferenceEngine:
    def __init__(self, model_name='roberta-large-mnli'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
    def check_entailment(self, premise, hypothesis):
        """Check if premise entails hypothesis"""
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        # Get entailment probability (class 2)
        entailment_prob = probs[0, 2].item()
        
        return {
            'entailment_probability': entailment_prob,
            'is_entailed': entailment_prob > 0.5
        }
        
    def infer_relations(self, text, knowledge_base):
        """Infer new relations from text using knowledge base"""
        inferred = []
        
        # Get existing relations
        existing_relations = knowledge_base.query_relations()
        
        for relation in existing_relations:
            # Create hypothesis using relation
            hypothesis = f"{relation['subject']} {relation['predicate']} {relation['object']}"
            
            # Check entailment
            result = self.check_entailment(text, hypothesis)
            
            if result['is_entailed']:
                inferred.append({
                    'relation': relation,
                    'confidence': result['entailment_probability']
                })
                
        return inferred
        
    def validate_inference(self, premise, conclusion, threshold=0.8):
        """Validate if a logical inference is valid"""
        result = self.check_entailment(premise, conclusion)
        
        return {
            'is_valid': result['entailment_probability'] > threshold,
            'confidence': result['entailment_probability']
        }
        
    def generate_hypotheses(self, text, templates):
        """Generate potential hypotheses from text using templates"""
        hypotheses = []
        
        # Extract entities (simplified)
        # In practice, use NER here
        words = text.split()
        
        for template in templates:
            # Fill template with entities
            for word in words:
                hypothesis = template.replace('[X]', word)
                
                # Check if hypothesis is entailed
                result = self.check_entailment(text, hypothesis)
                
                if result['is_entailed']:
                    hypotheses.append({
                        'hypothesis': hypothesis,
                        'confidence': result['entailment_probability']
                    })
                    
        return hypotheses