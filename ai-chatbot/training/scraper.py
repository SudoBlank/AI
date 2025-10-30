import requests
from bs4 import BeautifulSoup
import time
import random
import re
from urllib.parse import urljoin, urlparse
import json
from collections import deque
import os

class KnowledgeScraper:
    """Scrapes real websites to build training data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.visited_urls = set()
        self.knowledge_base = []
        
        # High-quality sources for diverse knowledge
        self.starting_urls = [
           
        ]
    
    def scrape_website(self, url, max_pages=100):
        """Scrape a website and extract meaningful content"""
        queue = deque([url])
        pages_scraped = 0
        
        while queue and pages_scraped < max_pages:
            current_url = queue.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            try:
                print(f"Scraping: {current_url}")
                response = self.session.get(current_url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract meaningful text
                title = self.clean_text(soup.find('title').get_text() if soup.find('title') else '')
                paragraphs = [self.clean_text(p.get_text()) for p in soup.find_all('p')]
                headings = [self.clean_text(h.get_text()) for h in soup.find_all(['h1', 'h2', 'h3'])]
                
                # Create knowledge entries
                knowledge = self.extract_knowledge(title, paragraphs, headings, current_url)
                self.knowledge_base.extend(knowledge)
                
                # Find new links
                if pages_scraped < max_pages // 2:  # Only follow links in first half
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(current_url, link['href'])
                        if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                            queue.append(full_url)
                
                self.visited_urls.add(current_url)
                pages_scraped += 1
                
                # Be respectful
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error scraping {current_url}: {e}")
                continue
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s.,!?;:]', '', text)  # Remove special chars but keep punctuation
        return text.strip()
    
    def is_valid_url(self, url):
        """Check if URL is valid for scraping"""
        parsed = urlparse(url)
        return (parsed.scheme in ['http', 'https'] and 
                len(parsed.path) > 1 and
                not any(blacklist in url for blacklist in [
                    'login', 'signup', 'logout', 'password', 'admin'
                ]))
    
    def extract_knowledge(self, title, paragraphs, headings, url):
        """Extract structured knowledge from page content"""
        knowledge = []
        
        # Create Q&A pairs from headings and paragraphs
        for heading in headings:
            if len(heading) > 10 and len(heading) < 200:
                # Find relevant paragraphs for this heading
                relevant_paras = [p for p in paragraphs if any(word in p.lower() for word in heading.lower().split()[:3])]
                
                if relevant_paras:
                    answer = ' '.join(relevant_paras[:2])  # Use first 2 relevant paragraphs
                    if len(answer) > 50:
                        knowledge.append({
                            'question': f"What is {heading}?",
                            'answer': answer,
                            'source': url,
                            'category': self.categorize_content(heading)
                        })
        
        # Extract facts from paragraphs
        for para in paragraphs:
            if len(para) > 100:
                # Simple fact extraction
                sentences = re.split(r'[.!?]', para)
                for sentence in sentences:
                    if self.is_factual(sentence) and len(sentence) > 20:
                        knowledge.append({
                            'question': self.generate_question(sentence),
                            'answer': sentence.strip(),
                            'source': url,
                            'category': 'general_knowledge'
                        })
        
        return knowledge
    
    def is_factual(self, sentence):
        """Check if sentence contains factual information"""
        factual_indicators = [
            'is a', 'are', 'was', 'were', 'can be', 'has', 'have', 
            'contains', 'includes', 'means', 'refers to', 'defined as'
        ]
        return any(indicator in sentence.lower() for indicator in factual_indicators)
    
    def generate_question(self, fact):
        """Generate a question from a factual statement"""
        words = fact.split()
        if len(words) > 3:
            return f"What is {words[0]}?" if len(words[0]) > 2 else f"Can you explain {fact.split(' is ')[0]}?"
        return f"Can you tell me about this?"
    
    def categorize_content(self, text):
        """Categorize content based on keywords"""
        categories = {
            'technology': ['python', 'javascript', 'programming', 'code', 'software', 'computer'],
            'science': ['physics', 'chemistry', 'biology', 'scientific', 'research'],
            'math': ['mathematics', 'equation', 'calculate', 'formula'],
            'history': ['historical', 'ancient', 'century', 'war', 'empire'],
            'ai': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning']
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'general_knowledge'
    
    def save_knowledge(self, filename='data/knowledge_base.json'):
        """Save scraped knowledge to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.knowledge_base)} knowledge entries to {filename}")
    
    def run_scraping_session(self, hours=24):
        """Run scraping for specified hours"""
        import time
        start_time = time.time()
        end_time = start_time + (hours * 3600)
        
        print(f"Starting {hours}-hour scraping session...")
        
        while time.time() < end_time:
            for url in self.starting_urls:
                if time.time() >= end_time:
                    break
                self.scrape_website(url, max_pages=50)
                self.save_knowledge()  # Save periodically
            
            print(f"Progress: {len(self.knowledge_base)} knowledge entries collected")
            time.sleep(300)  # Wait 5 minutes between rounds
        
        print(f"Scraping completed! Collected {len(self.knowledge_base)} knowledge entries.")