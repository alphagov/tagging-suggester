from src.utils.node import Node
import src.utils.directories as dirs
import gzip
import ijson
import os
import errno

class Tree:
    def __init__(self):
        self.nodes = {}
        taxons_path = os.path.join(dirs.processed_data_dir(), 'taxons.json.gz')
        if not os.path.exists(taxons_path) or not os.path.isfile(taxons_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"Taxon file doesn't exist, should be at: {taxons_path}")
        with gzip.open(taxons_path, mode='rt') as input_file:
            taxons = ijson.items(input_file, prefix='item')
            for taxon in taxons:
                node = Node(taxon, self.nodes)
                self.nodes[node.content_id] = node

    def find(self, taxon_content_id):
        """
        :param taxon_content_id: content_id of a taxon you want
        :return: Node instance of that taxon (if it exists)
        """
        return self.nodes[taxon_content_id]

    def apex_nodes(self):
        """
        :return: Node instance all top level taxons
        """
        apex_nodes = []
        for node in self.nodes.values():
            if node.is_apex():
                apex_nodes.append(node)
        return apex_nodes
