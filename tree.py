from node import Node
import gzip
import ijson
import os

class Tree:
    def __init__(self, datadir):
        self.nodes = {}
        taxons_path = os.path.join(datadir, 'taxons.json.gz')
        with gzip.open(taxons_path, mode='rt') as input_file:
            taxons = ijson.items(input_file, prefix='item')
            for taxon in taxons:
                node = Node(taxon, self.nodes)
                self.nodes[node.content_id] = node
    def find(self, taxon_content_id):
        return self.nodes[taxon_content_id]
    def apex_nodes(self):
        apex_nodes = []
        for node in self.nodes.values():
            if node.is_apex():
                apex_nodes.append(node)
        return apex_nodes