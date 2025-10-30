import networkx as nx
import json
from datetime import datetime
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_entity(self, entity_id, properties=None):
        """Add an entity node to the graph"""
        self.graph.add_node(
            entity_id,
            type='entity',
            properties=properties or {},
            created=datetime.now().isoformat()
        )
        
    def add_relation(self, subject, predicate, object_, properties=None):
        """Add a relation between entities"""
        # Add entities if they don't exist
        for entity in (subject, object_):
            if entity not in self.graph:
                self.add_entity(entity)
                
        # Add relation edge
        self.graph.add_edge(
            subject,
            object_,
            type='relation',
            predicate=predicate,
            properties=properties or {},
            created=datetime.now().isoformat()
        )
        
    def query_relations(self, subject=None, predicate=None, object_=None):
        """Query relations in the graph"""
        results = []
        
        for s, o, data in self.graph.edges(data=True):
            if data['type'] != 'relation':
                continue
                
            if subject and s != subject:
                continue
            if predicate and data['predicate'] != predicate:
                continue
            if object_ and o != object_:
                continue
                
            results.append({
                'subject': s,
                'predicate': data['predicate'],
                'object': o,
                'properties': data['properties']
            })
            
        return results
        
    def get_entity_relations(self, entity_id):
        """Get all relations involving an entity"""
        if entity_id not in self.graph:
            return []
            
        relations = []
        
        # Outgoing relations
        for _, o, data in self.graph.edges(entity_id, data=True):
            if data['type'] == 'relation':
                relations.append({
                    'direction': 'outgoing',
                    'subject': entity_id,
                    'predicate': data['predicate'],
                    'object': o
                })
                
        # Incoming relations
        for s, _, data in self.graph.in_edges(entity_id, data=True):
            if data['type'] == 'relation':
                relations.append({
                    'direction': 'incoming',
                    'subject': s,
                    'predicate': data['predicate'],
                    'object': entity_id
                })
                
        return relations
        
    def visualize(self, save_path=None):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(12, 8))
        
        # Draw graph
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1500,
            arrowsize=20
        )
        
        # Add relation labels
        edge_labels = {
            (s, o): d['predicate']
            for s, o, d in self.graph.edges(data=True)
            if d['type'] == 'relation'
        }
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels
        )
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def save(self, path):
        """Save graph to JSON file"""
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, path):
        """Load graph from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)