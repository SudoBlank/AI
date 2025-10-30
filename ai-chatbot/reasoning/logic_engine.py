import json
import re
from datetime import datetime

class LogicEngine:
    def __init__(self):
        self.rules = {}
        self.facts = set()
        
    def add_rule(self, if_clause, then_clause):
        """Add a logical rule"""
        # Convert rule to internal representation
        rule_id = len(self.rules)
        self.rules[rule_id] = {
            'if': if_clause,
            'then': then_clause,
            'created': datetime.now().isoformat()
        }
        return rule_id
        
    def add_fact(self, fact):
        """Add a fact to the knowledge base"""
        self.facts.add(tuple(fact))
        
    def query(self, pattern):
        """Query the knowledge base"""
        matches = []
        for fact in self.facts:
            if self._match_pattern(pattern, fact):
                matches.append(fact)
        return matches
        
    def infer(self):
        """Apply rules to infer new facts"""
        new_facts = set()
        
        # Keep inferring until no new facts are found
        while True:
            current_size = len(self.facts)
            
            # Try each rule
            for rule in self.rules.values():
                matches = self.query(rule['if'])
                for match in matches:
                    # Generate new fact from rule
                    new_fact = self._apply_rule(rule['then'], match)
                    if new_fact:
                        new_facts.add(tuple(new_fact))
                        
            # Add new facts
            self.facts.update(new_facts)
            
            # Stop if no new facts were added
            if len(self.facts) == current_size:
                break
                
        return new_facts
        
    def _match_pattern(self, pattern, fact):
        """Check if fact matches pattern"""
        if len(pattern) != len(fact):
            return False
            
        bindings = {}
        for p, f in zip(pattern, fact):
            if p.startswith('?'):  # Variable
                if p in bindings:
                    if bindings[p] != f:
                        return False
                else:
                    bindings[p] = f
            elif p != f:  # Literal
                return False
                
        return True
        
    def _apply_rule(self, conclusion, bindings):
        """Apply rule conclusion with variable bindings"""
        result = []
        for term in conclusion:
            if term.startswith('?'):
                if term in bindings:
                    result.append(bindings[term])
                else:
                    return None
            else:
                result.append(term)
        return result
        
    def save_knowledge(self, path):
        """Save rules and facts to file"""
        data = {
            'rules': self.rules,
            'facts': list(self.facts)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_knowledge(self, path):
        """Load rules and facts from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.rules = data['rules']
        self.facts = set(tuple(f) for f in data['facts'])