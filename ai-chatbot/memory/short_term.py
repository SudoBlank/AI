import json
from datetime import datetime
import os

class ShortTermMemory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.conversation = []
        
    def add_turn(self, user_input, bot_response):
        """Add a conversation turn"""
        turn = {
            'user_input': user_input,
            'bot_response': bot_response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation.append(turn)
        
        # Keep only recent turns
        if len(self.conversation) > self.max_turns:
            self.conversation = self.conversation[-self.max_turns:]
            
    def get_context(self, num_turns=None):
        """Get recent conversation context"""
        if num_turns is None:
            return self.conversation
        return self.conversation[-num_turns:]
        
    def clear(self):
        """Clear conversation history"""
        self.conversation = []