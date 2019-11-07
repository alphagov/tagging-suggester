from utils.content import *
from utils.tree import *

from data.representative_content import *
from models.apex_node_predictor import *
from models.branch_predictor import *

class TaggingSuggester:
    def train(self):
        """
        Processes data and trains models necessary to make tagging suggestions.
        Saves all files to disk
        """
        print("Loading data")
        content = Content()
        tree = Tree()
        print("Processing data")
        # RepresentativeContent(content, tree).generate()
        print("Training models")
        ApexNodePredictor().train(tree)
        BranchPredictor().train(content, tree)
        print("Done!")

    def predict(self, text):
        # This is pseudocode
        apex_nodes = ApexNodePredictor().predict(text)
        suggestions = []
        for apex_node in apex_nodes:
            suggestions += BranchPredictor.predict(apex_node)
        #sort suggestions somehow
        return suggestions